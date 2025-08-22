#!/usr/bin/env python3
"""
Single-drone RRT-Connect with narrow-passage sampling
  • Bridge sampling 
  • Gaussian sampling near obstacle boundaries
  • No smoothing; returns raw waypoints
  • Integrates with MultiDrone env by running with K=1

Run (example):
  python single_drone_narrow_rrt.py --env env_hard.yaml \
    --eta 0.8 --iters 120000 --goal-bias 0.1 \
    --p-bridge 0.4 --p-gauss 0.4 --sigma 1.2 \
    --seed 3 --verbose
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


# ---------------------------- Helpers ---------------------------- #

def infer_bounds_from_sim(sim) -> np.ndarray:
    """[[xmin,xmax],[ymin,ymax],[zmin,zmax]]"""
    if hasattr(sim, "environment") and isinstance(sim.environment, dict) and "bounds" in sim.environment:
        b = sim.environment["bounds"]
        return np.array([[b["x"][0], b["x"][1]],
                         [b["y"][0], b["y"][1]],
                         [b["z"][0], b["z"][1]]], dtype=float)
    return np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]], dtype=float)


@dataclass
class Node:
    q: np.ndarray         # (3,)
    parent: Optional[int] # index


class Tree:
    def __init__(self, root: np.ndarray):
        self.nodes: List[Node] = [Node(root.copy(), parent=None)]

    def add(self, q: np.ndarray, parent_idx: int) -> int:
        self.nodes.append(Node(q.copy(), parent=parent_idx))
        return len(self.nodes) - 1

    def nearest_idx(self, q: np.ndarray) -> int:
        q = q.reshape(3)
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(self.nodes):
            d = float(np.linalg.norm(q - n.q))
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def path_to_root(self, idx: int) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        while idx is not None:
            out.append(self.nodes[idx].q.copy())
            idx = self.nodes[idx].parent
        out.reverse()
        return out


# ---------------------------- Samplers ---------------------------- #

def sample_uniform(bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(bounds[0,0], bounds[0,1]),
                     rng.uniform(bounds[1,0], bounds[1,1]),
                     rng.uniform(bounds[2,0], bounds[2,1])], dtype=float)

def sample_gaussian_boundary(sim, bounds: np.ndarray, rng: np.random.Generator,
                             sigma: float, max_tries: int = 20) -> np.ndarray:
    """
    Gaussian sampling: pick a center u, then a neighbor v ~ N(u, sigma^2 I).
    If validity differs, return the VALID one (tends to lie near obstacle boundary).
    """
    for _ in range(max_tries):
        u = sample_uniform(bounds, rng)
        v = u + rng.normal(scale=sigma, size=3)
        v = np.clip(v, [bounds[0,0],bounds[1,0],bounds[2,0]],
                       [bounds[0,1],bounds[1,1],bounds[2,1]])
        if is_state_valid(sim, u) != is_state_valid(sim, v):
            return u if is_state_valid(sim, u) else v
    # fallback
    return sample_uniform(bounds, rng)

def sample_bridge(sim, bounds: np.ndarray, rng: np.random.Generator,
                  max_tries: int = 30) -> Optional[np.ndarray]:
    """
    Bridge test: draw two *invalid* samples a,b; if both invalid but midpoint m is valid → m is likely inside a narrow passage.
    """
    for _ in range(max_tries):
        a = sample_uniform(bounds, rng)
        b = sample_uniform(bounds, rng)
        if not is_state_valid(sim, a) and not is_state_valid(sim, b):
            m = 0.5*(a + b)
            if is_state_valid(sim, m):
                return m
    return None


# ---------------------------- Validity wrappers (K=1) ---------------------------- #

def cfg_from_point(p: np.ndarray) -> np.ndarray:
    """Turn (3,) into (1,3) config."""
    return np.asarray(p, float).reshape(1, 3)

def is_state_valid(sim, p: np.ndarray) -> bool:
    return bool(sim.is_valid(cfg_from_point(p)))

def edge_valid(sim, a: np.ndarray, b: np.ndarray) -> bool:
    return bool(sim.motion_valid(cfg_from_point(a), cfg_from_point(b)))

def in_goal(sim, p: np.ndarray) -> bool:
    return bool(sim.is_goal(cfg_from_point(p)))


# ---------------------------- RRT-Connect core ---------------------------- #

def steer(a: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
    d = float(np.linalg.norm(b - a))
    if d <= 1e-12:
        return a.copy()
    if d <= eta:
        return b.copy()
    return a + (eta / d) * (b - a)

def try_connect(sim, T: Tree, target: np.ndarray, eta: float) -> tuple[Optional[int], Optional[np.ndarray]]:
    """
    Extend T toward target; then greedily connect multiple steps (classic RRT-Connect).
    Returns (last_idx, last_q) if at least one step succeeded; else (None, None).
    """
    near_idx = T.nearest_idx(target)
    q_near = T.nodes[near_idx].q
    q_new = steer(q_near, target, eta)
    if not is_state_valid(sim, q_new) or not edge_valid(sim, q_near, q_new):
        return None, None

    last_idx = T.add(q_new, near_idx)
    last_q = q_new

    # greedy connect loop
    while True:
        q_step = steer(last_q, target, eta)
        if not is_state_valid(sim, q_step) or not edge_valid(sim, last_q, q_step):
            break
        last_idx = T.add(q_step, last_idx)
        last_q = q_step
        # stop if very close to target
        if float(np.linalg.norm(target - last_q)) < 0.25 * eta:
            break

    return last_idx, last_q


def narrow_rrt_connect(sim,
                       iters: int = 120000,
                       eta: float = 0.8,
                       goal_bias: float = 0.10,
                       p_bridge: float = 0.40,
                       p_gauss: float = 0.40,
                       sigma: float = 1.2,
                       seed: Optional[int] = None,
                       verbose: bool = True) -> Optional[List[np.ndarray]]:
    """
    Hybrid sampler RRT-Connect:
      - with prob goal_bias: sample goal center
      - with prob p_bridge:  bridge-test midpoint (if available)
      - with prob p_gauss:   Gaussian boundary sample
      - else:                uniform sample
    """
    rng = np.random.default_rng(seed)
    bounds = infer_bounds_from_sim(sim)

    start = np.asarray(sim.initial_configuration, float).reshape(-1,3)[0]
    goal  = np.asarray(sim.goal_positions, float).reshape(-1,3)[0]

    # Quick path
    if edge_valid(sim, start, goal):
        return [start.copy(), goal.copy()]

    Ta = Tree(start)
    Tb = Tree(goal)

    for k in range(1, iters + 1):
        # --- sampling policy ---
        r = rng.random()
        if r < goal_bias:
            q_rand = goal.copy()
        else:
            r2 = (r - goal_bias) / max(1e-12, (1.0 - goal_bias))
            if r2 < p_bridge:
                m = sample_bridge(sim, bounds, rng)
                q_rand = m if m is not None else sample_uniform(bounds, rng)
            elif r2 < p_bridge + p_gauss:
                q_rand = sample_gaussian_boundary(sim, bounds, rng, sigma=sigma)
            else:
                q_rand = sample_uniform(bounds, rng)

        # grow Ta toward q_rand
        idx_a, last_a = try_connect(sim, Ta, q_rand, eta)
        if idx_a is None:
            Ta, Tb = Tb, Ta
            continue

        # try to connect Tb to last_a
        idx_b, last_b = try_connect(sim, Tb, last_a, eta)
        if idx_b is not None and float(np.linalg.norm(last_a - last_b)) < 0.25 * eta:
            # reconstruct (start ... last_a) + (last_b ... goal) reversed
            path_a = Ta.path_to_root(idx_a)
            path_b = Tb.path_to_root(idx_b)
            left = path_a
            right = list(reversed(path_b))
            if np.allclose(left[-1], right[0], atol=1e-9):
                right = right[1:]
            joint = left + right

            # try to append exact goal if possible
            if not in_goal(sim, joint[-1]) and edge_valid(sim, joint[-1], goal):
                joint.append(goal.copy())

            if verbose:
                print(f"[RRT] success at iter {k} with {len(joint)} waypoints.")
            return joint

        # alternate trees
        Ta, Tb = Tb, Ta

    if verbose:
        print("[RRT] failed: iteration budget exhausted.")
    return None


# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Single-drone RRT-Connect with narrow-passage sampling")
    ap.add_argument("--env", type=str, default="single_drone_obs/env.yaml" ,help="Environment YAML (MultiDrone format)")
    ap.add_argument("--eta", type=float, default=0.8, help="Step size per extend")
    ap.add_argument("--iters", type=int, default=120000, help="Iteration budget")
    ap.add_argument("--goal-bias", type=float, default=0.10, help="Probability of sampling the goal")
    ap.add_argument("--p-bridge", type=float, default=0.40, help="Probability of bridge sampling")
    ap.add_argument("--p-gauss", type=float, default=0.40, help="Probability of Gaussian boundary sampling")
    ap.add_argument("--sigma", type=float, default=1.2, help="Gaussian sigma for boundary sampling")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--no-viz", action="store_true", help="Skip visualize_paths()")
    ap.add_argument("--verbose", action="store_true", help="Print status")
    args = ap.parse_args()

    # Your simulator
    from multi_drone import MultiDrone

    # Run as single-drone (K=1) regardless of the YAML’s list lengths
    sim = MultiDrone(num_drones=1, environment_file=args.env)

    path = narrow_rrt_connect(sim,
                              iters=args.iters,
                              eta=args.eta,
                              goal_bias=args.goal_bias,
                              p_bridge=args.p_bridge,
                              p_gauss=args.p_gauss,
                              sigma=args.sigma,
                              seed=args.seed,
                              verbose=args.verbose)

    if path is None:
        print("No path found.")
        return

    print(f"Found path with {len(path)} waypoints.")
    # If your visualizer expects a sequence of (K,3), lift to (1,3)
    k1_path = [np.asarray(p).reshape(1,3) for p in path]

    if not args.no_viz and hasattr(sim, "visualize_paths"):
        try:
            sim.visualize_paths(k1_path)
        except Exception as e:
            print(f"Visualization failed (continuing): {e}")


if __name__ == "__main__":
    main()
