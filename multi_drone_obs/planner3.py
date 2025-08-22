#!/usr/bin/env python3
"""
Step-synchronous multi-drone coordinator with micro-sampling and
**single-agent RRT escalation after N starved ticks**.

At each tick k:
  1) Each drone proposes a small step toward its goal (or stay put).
  2) If the straight step is invalid, try a micro-sampled first hop (tiny RRT, one-edge).
  3) If a drone has waited ≥ rrt_starve ticks, run a **single-agent RRT** with others frozen,
     cache the resulting waypoint list, and use the next waypoint as its proposal.
  4) Build a time-coupled conflict graph on proposals; pick a greedy MIS (priority = wait_age + β·dist
     with a big near-goal bonus) to move this tick; others wait.
  5) Commit chosen moves (jointly if valid, else serially). Repeat until all in goal or max-steps.

No smoothing; waypoints are the committed team configurations at each tick.

Run:
  python step_sync_coordinator.py --env env.yaml --K 12 \
    --eta 0.8 --max-steps 20000 --margin 0.0 \
    --starve 10 --micro-iters 600 --micro-bias 0.4 \
    --rrt-starve 25 --rrt-iters 30000 --rrt-eta 1.0 --rrt-goal-bias 0.2 \
    --seed 1 --verbose
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Set, Tuple, Optional

import numpy as np


# ---------------------------- Utils ---------------------------- #

def infer_bounds_from_sim(sim) -> np.ndarray:
    if hasattr(sim, "environment") and isinstance(sim.environment, dict) and "bounds" in sim.environment:
        bx = sim.environment["bounds"]
        return np.array(
            [
                [bx["x"][0], bx["x"][1]],
                [bx["y"][0], bx["y"][1]],
                [bx["z"][0], bx["z"][1]],
            ],
            dtype=float,
        )
    return np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]], dtype=float)


def will_pair_collide_info(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    r: float,
    margin: float = 0.0,
) -> Tuple[bool, float, float]:
    """
    Time-coupled closest approach for two synchronous straight-line motions over a tick.
    Returns (collides: bool, t_star in [0,1], d_min).
    """
    v = (b - a) - (d - c)
    w = (a - c)
    vv = float(np.dot(v, v))
    if vv == 0.0:
        t_star = 0.0
        d_min = float(np.linalg.norm(w))
    else:
        t_star = -float(np.dot(w, v)) / vv
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
    p = np.clip(p, [bounds[0, 0], bounds[1, 0], bounds[2, 0]], [bounds[0, 1], bounds[1, 1], bounds[2, 1]])
    return p


# ---------------------------- Micro-sampling (single-hop RRT) ---------------------------- #

def micro_first_hop(
    sim,
    current_cfg: np.ndarray,
    goal_i: np.ndarray,
    i: int,
    bounds: np.ndarray,
    eta: float,
    iters: int,
    goal_bias: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Try up to `iters` random samples (goal-biased) to find a single valid hop (<= eta)
    for drone i while all others are frozen. Returns the new position or None.
    """
    start = current_cfg[i]

    def steer(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        diff = p_to - p_from
        d = float(np.linalg.norm(diff))
        if d <= 1e-12:
            return p_from.copy()
        s = min(1.0, eta / d)
        return p_from + s * diff

    for _ in range(iters):
        if rng.random() < goal_bias:
            q_rand = goal_i
        else:
            q_rand = np.array(
                [
                    rng.uniform(bounds[0, 0], bounds[0, 1]),
                    rng.uniform(bounds[1, 0], bounds[1, 1]),
                    rng.uniform(bounds[2, 0], bounds[2, 1]),
                ],
                dtype=float,
            )
        q_new = steer(start, q_rand)
        cfg_a = current_cfg.copy()
        cfg_b = current_cfg.copy()
        cfg_b[i] = q_new
        if sim.motion_valid(cfg_a, cfg_b):
            return q_new
    return None


# ---------------------------- Single-agent RRT (others frozen) ---------------------------- #

class SingleAgentRRT:
    """RRT in 3D for ONE drone while all others remain fixed at current_cfg."""

    def __init__(
        self,
        sim,
        bounds: np.ndarray,
        current_cfg: np.ndarray,
        goal_pos: np.ndarray,
        moving_idx: int,
        eta: float = 1.0,
        goal_bias: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.sim = sim
        self.bounds = bounds.astype(float)
        self.current_cfg = current_cfg.copy()
        self.goal = goal_pos.astype(float).reshape(3)
        self.i = int(moving_idx)
        self.eta = float(eta)
        self.goal_bias = float(goal_bias)
        self.rng = np.random.default_rng(seed)
        self.nodes: List[np.ndarray] = [current_cfg[self.i].reshape(3).copy()]
        self.parents: List[Optional[int]] = [None]

    def sample(self) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return self.goal.copy()
        x = self.rng.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = self.rng.uniform(self.bounds[1, 0], self.bounds[1, 1])
        z = self.rng.uniform(self.bounds[2, 0], self.bounds[2, 1])
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def nearest_idx(self, q: np.ndarray) -> int:
        dists = [self.dist(n, q) for n in self.nodes]
        return int(np.argmin(dists))

    def steer(self, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        d = self.dist(p_from, p_to)
        if d <= 1e-12:
            return p_from.copy()
        if d <= self.eta:
            return p_to.copy()
        return p_from + (self.eta / d) * (p_to - p_from)

    def state_valid(self, p: np.ndarray) -> bool:
        cfg = self.current_cfg.copy()
        cfg[self.i] = p
        return bool(self.sim.is_valid(cfg))

    def edge_valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        cfg_a = self.current_cfg.copy()
        cfg_b = self.current_cfg.copy()
        cfg_a[self.i] = a
        cfg_b[self.i] = b
        return bool(self.sim.motion_valid(cfg_a, cfg_b))

    def in_goal(self, p: np.ndarray) -> bool:
        cfg = self.current_cfg.copy()
        cfg[self.i] = p
        return bool(self.sim.is_goal(cfg))

    def plan(self, max_iters: int) -> Optional[List[np.ndarray]]:
        if self.in_goal(self.nodes[0]):
            return [self.nodes[0].copy()]
        for _ in range(max_iters):
            q_rand = self.sample()
            idx = self.nearest_idx(q_rand)
            q_near = self.nodes[idx]
            q_new = self.steer(q_near, q_rand)
            if not self.state_valid(q_new):
                continue
            if not self.edge_valid(q_near, q_new):
                continue
            self.nodes.append(q_new.copy())
            self.parents.append(idx)
            new_idx = len(self.nodes) - 1
            if self.in_goal(q_new):
                return self._reconstruct(new_idx)
            if self.edge_valid(q_new, self.goal):
                self.nodes.append(self.goal.copy())
                self.parents.append(new_idx)
                return self._reconstruct(len(self.nodes) - 1)
        return None

    def _reconstruct(self, idx: int) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        while idx is not None:
            path.append(self.nodes[idx].copy())
            idx = self.parents[idx]
        path.reverse()
        return path


# ---------------------------- MIS selection ---------------------------- #

def build_conflict_graph(
    current: np.ndarray, proposals: np.ndarray, moving: List[int], r: float, margin: float
) -> Dict[int, Set[int]]:
    """Return adjacency among indices in 'moving' whose proposals collide if executed simultaneously."""
    adj: Dict[int, Set[int]] = {i: set() for i in moving}
    for a_idx in range(len(moving)):
        i = moving[a_idx]
        for b_idx in range(a_idx + 1, len(moving)):
            j = moving[b_idx]
            coll, _, _ = will_pair_collide_info(current[i], proposals[i], current[j], proposals[j], r, margin)
            if coll:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def maximal_independent_set(adj: Dict[int, Set[int]], scores: Dict[int, float]) -> List[int]:
    """Greedy MIS: sort by descending score, add node if no neighbor chosen."""
    chosen: List[int] = []
    blocked: Set[int] = set()
    order = sorted(adj.keys(), key=lambda i: (-scores.get(i, 0.0), i))
    for u in order:
        if u in blocked:
            continue
        if any((v in adj[u]) for v in chosen):
            continue
        chosen.append(u)
        blocked.update(adj[u])
    return chosen


# ---------------------------- Coordinator ---------------------------- #

def step_sync_plan(
    sim,
    eta: float = 0.8,
    max_steps: int = 5000 * 2,
    margin: float = 0.0,
    starve: int = 15,
    detour_scale: float = 1.0,
    beta_dist: float = 0.1,
    micro_iters: int = 400,
    micro_bias: float = 0.3,
    rrt_starve: int = 25,
    rrt_iters: int = 30000,
    rrt_eta: float = 1.0,
    rrt_goal_bias: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[np.ndarray]:
    """Run the step-synchronous loop and return a list of (K,3) configurations (no smoothing).

    Escalation: drones with wait_age >= rrt_starve get a single-agent RRT planned path (others frozen);
    we then propose its **next** waypoint each tick, still subject to MIS/time-coupled conflicts.
    """
    rng = np.random.default_rng(seed)
    bounds = infer_bounds_from_sim(sim)
    r = float(getattr(sim, "drone_radius", 0.3))

    current = np.asarray(sim.initial_configuration, float)
    goals = np.asarray(sim.goal_positions, float)
    K = current.shape[0]

    wait_age = np.zeros(K, dtype=int)
    path = [current.copy()]
    # cached per-drone RRT queues (remaining waypoints)
    rrt_queue: Dict[int, List[np.ndarray]] = {}

    for step in range(1, max_steps + 1):
        if hasattr(sim, "is_goal") and sim.is_goal(current):
            if verbose:
                print(f"Reached goal at step {step - 1} with {len(path)} waypoints.")
            break

        # 1) Baseline proposals (greedy toward goal)
        proposals = current.copy()
        for i in range(K):
            proposals[i] = greedy_step(current[i], goals[i], eta)

        # 2) Reject invalid straight steps; try micro-hop; if starved enough, try RRT queue
        candidates: List[int] = []
        for i in range(K):
            cfg = current.copy(); cfg[i] = proposals[i]
            if sim.motion_valid(current, cfg):
                candidates.append(i)
                continue

            # micro-hop
            p = micro_first_hop(sim, current, goals[i], i, bounds, eta, micro_iters, micro_bias, rng)
            if p is not None:
                cfg2 = current.copy(); cfg2[i] = p
                if sim.motion_valid(current, cfg2):
                    proposals[i] = p
                    candidates.append(i)
                    continue

            # escalate to single-agent RRT if starved
            if wait_age[i] >= rrt_starve:
                # ensure we have a queue
                if i not in rrt_queue or len(rrt_queue[i]) == 0:
                    rrt = SingleAgentRRT(sim=sim, bounds=bounds, current_cfg=current, goal_pos=goals[i],
                                          moving_idx=i, eta=rrt_eta, goal_bias=rrt_goal_bias,
                                          seed=int(rng.integers(1 << 30)))
                    i_path = rrt.plan(max_iters=rrt_iters)
                    if i_path is not None and len(i_path) >= 2:
                        rrt_queue[i] = i_path[1:]  # drop the current position
                    else:
                        rrt_queue[i] = []
                # try to use the next queued waypoint
                if i in rrt_queue and len(rrt_queue[i]) > 0:
                    p = rrt_queue[i][0]
                    cfg3 = current.copy(); cfg3[i] = p
                    if sim.motion_valid(current, cfg3):
                        proposals[i] = p
                        candidates.append(i)
                    else:
                        # drop unusable step; we'll re-evaluate next tick
                        rrt_queue[i].pop(0)

        # 3) If nobody has a valid move, try micro-hop (then detour) for most-starved
        if not candidates:
            idx = int(np.argmax(wait_age))
            p = micro_first_hop(sim, current, goals[idx], idx, bounds, eta * detour_scale, micro_iters, micro_bias, rng)
            if p is None:
                p = random_detour(current[idx], bounds, detour_scale * eta, rng)
            cfg = current.copy(); cfg[idx] = p
            if sim.motion_valid(current, cfg):
                proposals[idx] = p
                candidates.append(idx)
                if verbose:
                    print(f"[tick {step}] Forced micro/detour proposal for drone {idx}.")
            else:
                if verbose:
                    print(f"[tick {step}] No feasible moves; all wait.")
                wait_age += 1
                path.append(current.copy())
                continue

        # 4) Build conflict graph among candidates using time-coupled check
        adj = build_conflict_graph(current, proposals, candidates, r=r, margin=margin)

        # 5) Priorities (wait_age first, then remaining distance; boost near-goal)
        dists = np.linalg.norm(goals - current, axis=1)
        near = dists < (1.5 * eta)
        scores = {i: (wait_age[i] + (100.0 if near[i] else 0.0) + beta_dist * dists[i]) for i in candidates}

        # 6) MIS selection
        chosen = maximal_independent_set(adj, scores)

        # 7) Apply chosen moves (jointly if valid; else serially and track who actually moved)
        next_cfg = current.copy()
        moved_list: List[int] = []

        # Try joint
        for i in chosen:
            next_cfg[i] = proposals[i]
        if not sim.is_valid(next_cfg):
            if verbose:
                print(f"[tick {step}] Joint next_cfg invalid; trying serial application of chosen.")
            tmp = current.copy()
            moved_list = []
            for i in chosen:
                cfg = tmp.copy(); cfg[i] = proposals[i]
                if sim.motion_valid(tmp, cfg):
                    tmp = cfg
                    moved_list.append(i)
            if not moved_list:
                if verbose:
                    print(f"[tick {step}] Even serial application failed; all wait.")
                wait_age += 1
                path.append(current.copy())
                continue
            next_cfg = tmp
        else:
            moved_list = list(chosen)

        # Commit
        path.append(next_cfg.copy())
        moved = set(moved_list)

        # Update wait ages
        for i in range(K):
            wait_age[i] = 0 if i in moved else wait_age[i] + 1

        # Advance and maintain RRT queues (pop used steps)
        for i in moved:
            if i in rrt_queue and len(rrt_queue[i]) > 0 and np.allclose(rrt_queue[i][0], next_cfg[i]):
                rrt_queue[i].pop(0)
                if len(rrt_queue[i]) == 0:
                    del rrt_queue[i]
        current = next_cfg

        if verbose:
            avg_rem = float(np.mean(np.linalg.norm(goals - current, axis=1)))
            print(f"[tick {step}] moved {len(moved)} drones; remaining avg dist={avg_rem:.2f}")

    return path


# ---------------------------- CLI ---------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Step-synchronous multi-drone coordinator (micro + RRT escalation)")
    ap.add_argument("--env", type=str, default="multi_drone_obs/env4.yaml", help="Environment YAML file")
    ap.add_argument("--K", type=int, default=None, help="Number of drones (defaults to YAML)")
    ap.add_argument("--eta", type=float, default=0.8, help="Max step length per tick (meters)")
    ap.add_argument("--max-steps", type=int, default=5000, help="Maximum ticks to run")
    ap.add_argument("--margin", type=float, default=0.0, help="Extra safety margin added to 2*r in pairwise checks")
    ap.add_argument("--starve", type=int, default=15, help="Ticks of waiting before micro-hop/detour is tried")
    ap.add_argument("--detour-scale", type=float, default=1.0, help="Detour step as multiple of eta when starved")
    ap.add_argument("--beta-dist", type=float, default=0.1, help="Weight of remaining distance in MIS priority")
    ap.add_argument("--micro-iters", type=int, default=400, help="Trials for micro first-hop sampler per attempt")
    ap.add_argument("--micro-bias", type=float, default=0.1, help="Goal bias for micro first-hop sampler")
    ap.add_argument("--rrt-starve", type=int, default=25, help="Ticks of waiting before escalating to single-agent RRT")
    ap.add_argument("--rrt-iters", type=int, default=30000, help="Iteration budget for single-agent RRT")
    ap.add_argument("--rrt-eta", type=float, default=1.0, help="Step size for single-agent RRT")
    ap.add_argument("--rrt-goal-bias", type=float, default=0.2, help="Goal bias for single-agent RRT")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ap.add_argument("--verbose", action="store_true", help="Print per-tick info")
    ap.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = ap.parse_args()

    from multi_drone import MultiDrone

    sim = MultiDrone(num_drones=args.K, environment_file=args.env)

    path = step_sync_plan(
        sim,
        eta=args.eta,
        max_steps=args.max_steps,
        margin=args.margin,
        starve=args.starve,
        detour_scale=args.detour_scale,
        beta_dist=args.beta_dist,
        micro_iters=args.micro_iters,
        micro_bias=args.micro_bias,
        rrt_starve=args.rrt_starve,
        rrt_iters=args.rrt_iters,
        rrt_eta=args.rrt_eta,
        rrt_goal_bias=args.rrt_goal_bias,
        seed=args.seed,
        verbose=args.verbose,
    )

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
