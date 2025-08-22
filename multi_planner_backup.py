#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Use the same import style as your example
from Planner.multi_drone import MultiDrone


# -------------------------- Small helpers --------------------------

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _get_bounds_2k(sim: MultiDrone) -> np.ndarray:
    """
    Return bounds as shape (2, K): [[min...],[max...]]
    Accepts (2, K) or (K, 2) on sim._bounds.
    """
    if not hasattr(sim, "_bounds"):
        raise RuntimeError("MultiDrone missing _bounds; please expose [[min...],[max...]].")
    b = np.asarray(sim._bounds, dtype=float)
    if b.ndim != 2:
        raise RuntimeError(f"Unexpected bounds shape {b.shape}")
    if b.shape[0] == 2:
        return b
    if b.shape[1] == 2:
        return b.T
    raise RuntimeError(f"Bounds must be (2,K) or (K,2); got {b.shape}")

def _advance_along(polyline: List[np.ndarray], step_len: float, start_pos: np.ndarray) -> np.ndarray:
    """
    Move 'step_len' forward along 'polyline', starting from 'start_pos'.
    """
    pos = start_pos.copy()
    remain = step_len
    for k in range(1, len(polyline)):
        a, b = polyline[k - 1], polyline[k]
        seg = b - a
        L = _norm(seg)
        if L < 1e-12:
            continue
        # where is pos along the segment a->b?
        t = 0.0 if _norm(pos - a) < 1e-9 else float(((pos - a) @ seg) / (L * L))
        t = min(1.0, max(0.0, t))
        p_on = a + t * seg
        ahead = _norm(b - p_on)
        if ahead >= remain:
            return p_on + (seg / L) * remain
        remain -= ahead
        pos = b.copy()
    return pos

# -------------------------- One-drone RRT-Connect --------------------------

@dataclass
class _Node:
    q: np.ndarray
    parent: Optional[int]

class _Tree:
    def __init__(self, root: np.ndarray):
        root = np.asarray(root, float).reshape(-1)
        self.nodes: List[_Node] = [_Node(root.copy(), None)]
        self.points: List[np.ndarray] = [root.copy()]

    def add(self, q: np.ndarray, parent: int) -> int:
        q = np.asarray(q, float).reshape(-1)
        self.nodes.append(_Node(q.copy(), parent))
        self.points.append(q.copy())
        return len(self.nodes) - 1

    def nearest_idx(self, q: np.ndarray) -> int:
        Q = np.asarray(self.points, dtype=float)   # (n, D)
        q = np.asarray(q, dtype=float).reshape(1, -1)
        d = np.linalg.norm(Q - q, axis=1)
        return int(np.argmin(d))

    def path_to(self, idx: int) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        i = int(idx)
        while i is not None:
            out.append(self.nodes[i].q.copy())
            i = self.nodes[i].parent
        out.reverse()
        return out

def _edge_ok_group(sim: MultiDrone, cur_group: np.ndarray, drone_i: int, a: np.ndarray, b: np.ndarray) -> bool:
    """
    Validate moving ONLY drone_i straight from a->b with others fixed.
    Uses sim.motion_valid(start_group, end_group).
    """
    start = cur_group.astype(np.float32).copy()
    end   = cur_group.astype(np.float32).copy()
    start[drone_i] = np.asarray(a, float).astype(np.float32)
    end[drone_i]   = np.asarray(b, float).astype(np.float32)
    return bool(sim.motion_valid(start, end))

def _steer(a: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    d = b - a
    L = _norm(d)
    if L <= eta:
        return b.copy()
    return a + d * (eta / L)

def _sample(bounds_2k: np.ndarray, rng: np.random.Generator, like: np.ndarray) -> np.ndarray:
    """
    Sample a point with the SAME dimension as vector `like`.
    If bounds offer fewer dims K < D, pad the extra dims with current coords in `like`.
    """
    like = np.asarray(like, float).reshape(-1)
    D = like.size
    B = np.asarray(bounds_2k, float)  # (2, K)
    K = B.shape[1]
    if K >= D:
        lo = B[0, :D]
        hi = B[1, :D]
    else:
        lo = np.concatenate([B[0, :], like[K:]])
        hi = np.concatenate([B[1, :], like[K:]])
    return rng.uniform(lo, hi)

def rrt_connect_step(
    sim: MultiDrone,
    cur_group: np.ndarray,     # (N, D) current positions of all drones
    drone_i: int,              # active drone index
    start: np.ndarray,         # (D,) active current
    goal: np.ndarray,          # (D,) active goal (center)
    eta: float = 0.8,
    max_iterations: int = 4000,
    goal_radius: float = 1.0,
    seed: Optional[int] = None
) -> Optional[List[np.ndarray]]:
    """
    RRT-Connect for ONE drone while others are frozen.
    Edge checks use sim.motion_valid on the group.
    """
    start = np.asarray(start, float).reshape(-1)
    goal  = np.asarray(goal,  float).reshape(-1)
    assert start.shape == goal.shape, f"start {start.shape} vs goal {goal.shape}"

    rng = np.random.default_rng(seed)
    bounds_2k = _get_bounds_2k(sim)

    # straight-line quick check
    if _edge_ok_group(sim, cur_group, drone_i, start, goal):
        if _norm(start - goal) <= goal_radius:
            return [start.copy(), goal.copy()]
        # even if > goal_radius, straight line is fine; caller will step partially
        return [start.copy(), goal.copy()]

    Ta = _Tree(start)
    Tb = _Tree(goal)

    for _ in range(max_iterations):
        q_rand = _sample(bounds_2k, rng, start)
        # extend Ta toward q_rand
        ia = Ta.nearest_idx(q_rand)
        qa = Ta.nodes[ia].q
        qa_new = _steer(qa, q_rand, eta)
        if _edge_ok_group(sim, cur_group, drone_i, qa, qa_new):
            ia_new = Ta.add(qa_new, ia)
            # connect Tb toward qa_new
            while True:
                ib = Tb.nearest_idx(qa_new)
                qb = Tb.nodes[ib].q
                qb_new = _steer(qb, qa_new, eta)
                if not _edge_ok_group(sim, cur_group, drone_i, qb, qb_new):
                    break
                ib_new = Tb.add(qb_new, ib)
                if _norm(qb_new - qa_new) <= eta:
                    # stitch
                    path_a = Ta.path_to(ia_new)
                    path_b = Tb.path_to(ib_new)
                    path_b.reverse()
                    if np.allclose(path_a[-1], path_b[0], atol=1e-9):
                        path_b = path_b[1:]
                    return path_a + path_b
                ib = ib_new
        # swap roles
        Ta, Tb = Tb, Ta

    # last small hop if within eta and valid
    if _norm(start - goal) <= eta and _edge_ok_group(sim, cur_group, drone_i, start, goal):
        return [start.copy(), goal.copy()]
    return None

# -------------------------- Online one-at-a-time loop --------------------------

@dataclass
class _State:
    t: float
    pos: np.ndarray  # (N, D)

def online_one_at_a_time(
    sim: MultiDrone,
    speed: float = 2.0,
    dt: float = 0.1,
    eta: float = 0.8,
    max_rrt_iters: int = 4000,
    priority: Optional[List[int]] = None,
    seed: Optional[int] = None,
    max_steps: int = 200000
) -> List[np.ndarray]:
    """
    Moves one drone per time step. For the active drone, plans with RRT-Connect
    from its CURRENT position to its goal (others fixed), then advances a small
    step (speed*dt) along the planned local path.
    Returns: frames = [ (N,D) float32 arrays ] for sim.visualize_paths(frames).
    """
    rng = np.random.default_rng(seed)

    cur = np.asarray(sim.initial_configuration, float)  # (N,D)
    goals = np.asarray(sim.goal_positions, float)       # (N,D)
    if cur.ndim != 2 or goals.ndim != 2:
        raise RuntimeError("Expected initial_configuration and goal_positions as (N,D) arrays.")
    N, D = cur.shape
    if goals.shape != (N, D):
        raise RuntimeError(f"Goal shape {goals.shape} incompatible with start shape {(N, D)}.")

    # radii: try to read from sim; else from YAML later; else default
    if hasattr(sim, "goal_radii"):
        goal_r = np.asarray(getattr(sim, "goal_radii"), float).reshape(-1)
    elif hasattr(sim, "_goal_radii"):
        goal_r = np.asarray(getattr(sim, "_goal_radii"), float).reshape(-1)
    else:
        goal_r = np.full(N, 0.5, float)

    order = list(range(N)) if priority is None else list(priority)
    assert len(order) == N and set(order) == set(range(N)), "priority must permute [0..N-1]"

    frames: List[np.ndarray] = [cur.astype(np.float32)]
    done = np.array([_norm(cur[i] - goals[i]) <= goal_r[i] for i in range(N)], dtype=bool)
    step_len = float(speed * dt)

    idx_order = 0
    t = 0.0
    steps = 0

    # sanity: whole-group start is valid
    if not sim.is_valid(cur.astype(np.float32)):
        raise RuntimeError("Initial configuration is invalid per MultiDrone.is_valid().")

    while not bool(np.all(done)) and steps < max_steps:
        steps += 1
        t += dt
        i = order[idx_order]
        idx_order = (idx_order + 1) % N

        if done[i]:
            frames.append(cur.astype(np.float32))
            continue

        # Plan locally for drone i, keeping others fixed
        path = rrt_connect_step(
            sim=sim,
            cur_group=cur,
            drone_i=i,
            start=cur[i],
            goal=goals[i],
            eta=eta,
            max_iterations=max_rrt_iters,
            goal_radius=float(goal_r[i]),
            seed=int(rng.integers(0, 2**31 - 1))
        )

        if path is None or len(path) < 2:
            # no move possible this turn
            frames.append(cur.astype(np.float32))
            continue

        # Advance a small step along the freshly planned path
        next_pos = _advance_along(path, step_len, start_pos=cur[i])

        # Belt & braces: validate the group motion for this step
        start = cur.astype(np.float32).copy()
        end   = cur.astype(np.float32).copy()
        end[i] = next_pos.astype(np.float32)
        if not sim.motion_valid(start, end):
            # wait if the short hop is not valid
            frames.append(cur.astype(np.float32))
            continue

        # Commit
        cur[i] = next_pos
        if _norm(cur[i] - goals[i]) <= goal_r[i]:
            done[i] = True

        frames.append(cur.astype(np.float32))

    return frames

# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="Online one-at-a-time MultiDrone planner (RRT per active drone).")
    ap.add_argument("--env", default="multi_drone_obs/env.yaml")
    ap.add_argument("--speed", type=float, default=2.0, help="Speed (m/s)")
    ap.add_argument("--dt", type=float, default=0.1, help="Step time (s)")
    ap.add_argument("--eta", type=float, default=0.8, help="RRT step size")
    ap.add_argument("--rrt-iters", type=int, default=4000, help="Max RRT iterations per step")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=200000)
    args = ap.parse_args()

    # Get num_drones from YAML so we can construct MultiDrone like your example
    with open(args.env, "r") as f:
        cfg = yaml.safe_load(f)
    init_cfg = np.asarray(cfg["initial_configuration"], float)
    num_drones = int(init_cfg.shape[0])

    sim = MultiDrone(num_drones=num_drones, environment_file=args.env)

    frames = online_one_at_a_time(
        sim,
        speed=args.speed,
        dt=args.dt,
        eta=args.eta,
        max_rrt_iters=args.rrt_iters,
        priority=None,
        seed=args.seed,
        max_steps=args.max_steps
    )

    # Visualize with your API
    sim.visualize_paths(frames)


if __name__ == "__main__":
    main()
