#!/usr/bin/env python3
"""
Step-synchronous multi-drone coordinator (distributed feel, central referee).

At each tick k:
  1) Each drone proposes a small step toward its goal (or stay put).
  2) Reject proposals that collide with static world + stationary drones (via sim.motion_valid).
  3) Build a conflict graph among *accepted* proposals using time-coupled closest approach.
  4) Pick a maximal independent set (MIS) by priority (wait_age, distance-to-goal).
  5) Execute only the MIS moves; others wait. Repeat until all in goal or max-steps.

If no one can move on a tick, we let the most-starved drones propose a random detour step.
No smoothing; waypoints are the committed team configurations at each tick.

Run:
  python step_sync_coordinator.py --env env2.yaml --K 12 --eta 0.8 --max-steps 5000 --margin 0.1 \
    --starve 15 --detour-scale 1.0 --seed 1
"""
from __future__ import annotations
import argparse
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

# ---------------------------- Utils ---------------------------- #

def infer_bounds_from_sim(sim) -> np.ndarray:
    if hasattr(sim, "environment") and isinstance(sim.environment, dict) and "bounds" in sim.environment:
        bx = sim.environment["bounds"]
        return np.array([[bx["x"][0], bx["x"][1]],
                         [bx["y"][0], bx["y"][1]],
                         [bx["z"][0], bx["z"][1]]], dtype=float)
    return np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]], dtype=float)


def will_pair_collide_info(a: np.ndarray, b: np.ndarray,
                           c: np.ndarray, d: np.ndarray,
                           r: float, margin: float = 0.0) -> Tuple[bool, float, float]:
    """Time-coupled closest approach for two synchronous straight-line motions over a tick.
    Returns (collides: bool, t_star in [0,1], d_min)."""
    v = (b - a) - (d - c)
    w = (a - c)
    vv = float(np.dot(v, v))
    if vv == 0.0:
        t_star = 0.0
        d_min = float(np.linalg.norm(w))
    else:
        t_star = - float(np.dot(w, v)) / vv
        t_star = max(0.0, min(1.0, t_star))
        d_min = float(np.linalg.norm(w + t_star * v))
    return (d_min < 2.0 * (r + margin)), t_star, d_min


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------- Proposals ---------------------------- #

def greedy_step(current: np.ndarray, goal: np.ndarray, eta: float) -> np.ndarray:
    d = distance(current, goal)
    if d <= 1e-12:
        return current.copy()
    s = min(1.0, eta / d)
    return current + s * (goal - current)


def random_detour(current: np.ndarray, bounds: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    # random direction on sphere, length <= eta, clamp to bounds
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v = np.array([1.0, 0.0, 0.0])
        n = 1.0
    step = (eta / n) * v
    p = current + step
    # clamp to bounds box
    p = np.clip(p, [bounds[0,0], bounds[1,0], bounds[2,0]], [bounds[0,1], bounds[1,1], bounds[2,1]])
    return p


# ---------------------------- MIS selection ---------------------------- #

def build_conflict_graph(current: np.ndarray, proposals: np.ndarray, moving: List[int], r: float, margin: float) -> Dict[int, Set[int]]:
    """Return adjacency among indices in 'moving' whose proposals collide if executed simultaneously."""
    adj: Dict[int, Set[int]] = {i: set() for i in moving}
    for idx_i in range(len(moving)):
        i = moving[idx_i]
        for idx_j in range(idx_i + 1, len(moving)):
            j = moving[idx_j]
            coll, _, _ = will_pair_collide_info(current[i], proposals[i], current[j], proposals[j], r, margin)
            if coll:
                adj[i].add(j); adj[j].add(i)
    return adj


def maximal_independent_set(adj: Dict[int, Set[int]], scores: Dict[int, float]) -> List[int]:
    """Greedy MIS: sort by descending score, add node if no neighbor chosen."""
    chosen: List[int] = []
    blocked: Set[int] = set()
    order = sorted(adj.keys(), key=lambda i: (-scores.get(i, 0.0), i))
    for u in order:
        if u in blocked:
            continue
        # check if u conflicts with any already chosen
        if any((v in adj[u]) for v in chosen):
            continue
        chosen.append(u)
        blocked.update(adj[u])
    return chosen


# ---------------------------- Coordinator ---------------------------- #

def step_sync_plan(sim,
                   eta: float = 0.8,
                   max_steps: int = 5000,
                   margin: float = 0.0,
                   starve: int = 15,
                   detour_scale: float = 1.0,
                   beta_dist: float = 0.1,
                   seed: Optional[int] = None,
                   verbose: bool = True) -> List[np.ndarray]:
    """Run the step-synchronous loop and return a list of (K,3) configurations (no smoothing)."""
    rng = np.random.default_rng(seed)
    bounds = infer_bounds_from_sim(sim)
    r = float(getattr(sim, "drone_radius", 0.3))

    current = np.asarray(sim.initial_configuration, float)
    goals   = np.asarray(sim.goal_positions, float)
    K = current.shape[0]

    wait_age = np.zeros(K, dtype=int)
    path = [current.copy()]

    for step in range(1, max_steps + 1):
        # If done, break
        if hasattr(sim, "is_goal") and sim.is_goal(current):
            if verbose:
                print(f"Reached goal at step {step-1} with {len(path)} waypoints.")
            break

        # 1) Baseline proposals (greedy toward goal)
        proposals = current.copy()
        for i in range(K):
            # If already very close, snap to goal
            proposals[i] = greedy_step(current[i], goals[i], eta)

        # 2) Reject proposals that hit static obstacles / stationary drones
        candidates: List[int] = []
        for i in range(K):
            cfg = current.copy(); cfg[i] = proposals[i]
            if sim.motion_valid(current, cfg):
                candidates.append(i)
            # else keep it out; it will be considered as 'wait'

        # 3) If nobody has a valid greedy move, let starved drones try detours
        if not candidates:
            # pick all with max wait_age (>= starve) else the top 1-2
            max_age = int(wait_age.max())
            picks = [i for i in range(K) if wait_age[i] == max_age and max_age >= starve]
            if not picks:
                picks = [int(np.argmax(wait_age))]
            changed = False
            for i in picks:
                p = random_detour(current[i], bounds, detour_scale * eta, rng)
                cfg = current.copy(); cfg[i] = p
                if sim.motion_valid(current, cfg):
                    proposals[i] = p
                    candidates.append(i)
                    changed = True
            if verbose and changed:
                print(f"[tick {step}] Detour proposals accepted for drones: {candidates}")

        # 4) Build conflict graph among candidates using time-coupled check
        adj = build_conflict_graph(current, proposals, candidates, r=r, margin=margin)

        # 5) Priorities (wait_age first, then remaining distance)
        dists = np.linalg.norm(goals - current, axis=1)
        scores = {i: (wait_age[i] + beta_dist * dists[i]) for i in candidates}

        # 6) MIS selection
        chosen = maximal_independent_set(adj, scores)

        # 7) If MIS empty, everyone waits this tick (avoid infinite loop by forcing one mover)
        if not chosen:
            # pick the most starved with a valid proposal if any, else the most starved overall and make a tiny detour
            idx = int(np.argmax(wait_age))
            cfg = current.copy(); cfg[idx] = random_detour(current[idx], bounds, 0.5 * eta, rng)
            if sim.motion_valid(current, cfg):
                proposals[idx] = cfg[idx]
                chosen = [idx]
            else:
                if verbose:
                    print(f"[tick {step}] No feasible moves; all wait.")
                wait_age += 1
                path.append(current.copy())
                continue

        # 8) Apply chosen moves
        next_cfg = current.copy()
        for i in chosen:
            next_cfg[i] = proposals[i]

        # Final safety: validate joint move vs world (moving-moving already checked pairwise)
        if not sim.is_valid(next_cfg):
            # if invalid (e.g., bounds), fall back to serial apply
            if verbose:
                print(f"[tick {step}] Joint next_cfg invalid; trying serial application of chosen.")
            ok_any = False
            tmp = current.copy()
            for i in chosen:
                cfg = tmp.copy(); cfg[i] = proposals[i]
                if sim.motion_valid(tmp, cfg):
                    tmp = cfg
                    ok_any = True
            if not ok_any:
                # nobody could be applied serially; wait and retry next tick
                if verbose:
                    print(f"[tick {step}] Even serial application failed; all wait.")
                wait_age += 1
                path.append(current.copy())
                continue
            next_cfg = tmp

        # Commit
        path.append(next_cfg.copy())
        # Update wait ages
        moved = set(chosen)
        for i in range(K):
            wait_age[i] = 0 if i in moved else wait_age[i] + 1
        current = next_cfg

        if verbose:
            print(f"[tick {step}] moved {len(chosen)} drones; remaining avg dist={np.mean(np.linalg.norm(goals-current,axis=1)):.2f}")

    return path


# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Step-synchronous multi-drone coordinator (distributed proposals)")
    ap.add_argument("--env", type=str, default="multi_drone_obs/env.yaml", help="Environment YAML file")
    ap.add_argument("--K", type=int, default=None, help="Number of drones (defaults to YAML)")
    ap.add_argument("--eta", type=float, default=0.8, help="Max step length per tick (meters)")
    ap.add_argument("--max-steps", type=int, default=10000 , help="Maximum ticks to run")
    ap.add_argument("--margin", type=float, default=0.0, help="Extra safety margin added to 2*r in pairwise checks")
    ap.add_argument("--starve", type=int, default=15, help="Ticks of waiting before detour proposals are allowed")
    ap.add_argument("--detour-scale", type=float, default=1.0, help="Detour step as multiple of eta when starved")
    ap.add_argument("--beta-dist", type=float, default=0.1, help="Weight of remaining distance in MIS priority")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--verbose", action="store_true", help="Print per-tick info")
    ap.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = ap.parse_args()

    from multi_drone import MultiDrone

    sim = MultiDrone(num_drones=args.K, environment_file=args.env)

    path = step_sync_plan(sim,
                          eta=args.eta,
                          max_steps=args.max_steps,
                          margin=args.margin,
                          starve=args.starve,
                          detour_scale=args.detour_scale,
                          beta_dist=args.beta_dist,
                          seed=args.seed,
                          verbose=args.verbose)

    print(f"Produced {len(path)} waypoints.")
    if hasattr(sim, "is_goal"):
        print("Goal reached?", sim.is_goal(path[-1]))

    if not args.no_viz and hasattr(sim, "visualize_paths"):
        try:
            sim.visualize_paths(path)
        except Exception as e:
            print(f"Visualization failed (continuing): {e}")

if __name__ == "__main__":
    main()
