#!/usr/bin/env python3

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ------------------------- Small helpers ------------------------- #

def infer_bounds_from_sim(sim) -> np.ndarray:
    """Return [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."""
    if hasattr(sim, "environment") and isinstance(sim.environment, dict) and "bounds" in sim.environment:
        bx = sim.environment["bounds"]
        return np.array([[bx["x"][0], bx["x"][1]],
                         [bx["y"][0], bx["y"][1]],
                         [bx["z"][0], bx["z"][1]]], dtype=float)
    return np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]], dtype=float)


def to_cfg(p: np.ndarray) -> np.ndarray:
    """Convert (3,) to simulator config shape (1,3)."""
    return p.reshape(1, 3)


# ------------------------- RRT data structures ------------------------- #

@dataclass
class Node:
    p: np.ndarray      # (3,)
    parent: Optional[int]


class RRT:
    def __init__(self, sim, eta: float, goal_bias: float, seed: Optional[int] = None):
        self.sim = sim
        self.eta = float(eta)
        self.goal_bias = float(goal_bias)
        self.rng = np.random.default_rng(seed)
        self.bounds = infer_bounds_from_sim(sim)
        self.goal_center = np.asarray(sim.goal_positions, float).reshape(-1, 3)[0]

        start = np.asarray(sim.initial_configuration, float)
        if start.shape[0] != 1:
            raise ValueError("This RRT is for a SINGLE drone. Set num_drones=1 in MultiDrone.")
        self.start = start.reshape(3)

        self.nodes: List[Node] = [Node(self.start.copy(), parent=None)]

    # ---- geometry ---- #
    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def sample(self) -> np.ndarray:
        """Goal-biased uniform sample inside bounds."""
        if self.rng.random() < self.goal_bias:
            return self.goal_center.copy()
        x = self.rng.uniform(self.bounds[0, 0], self.bounds[0, 1])
        y = self.rng.uniform(self.bounds[1, 0], self.bounds[1, 1])
        z = self.rng.uniform(self.bounds[2, 0], self.bounds[2, 1])
        return np.array([x, y, z], dtype=float)

    def nearest_idx(self, q: np.ndarray) -> int:
        dists = [self.dist(n.p, q) for n in self.nodes]
        return int(np.argmin(dists))

    def steer(self, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        """Step from p_from toward p_to by at most eta."""
        d = self.dist(p_from, p_to)
        if d <= 1e-12:
            return p_from.copy()
        if d <= self.eta:
            return p_to.copy()
        direction = (p_to - p_from) / d
        return p_from + self.eta * direction

    # ---- validity wrappers ---- #
    def state_valid(self, p: np.ndarray) -> bool:
        return bool(self.sim.is_valid(to_cfg(p)))

    def edge_valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(self.sim.motion_valid(to_cfg(a), to_cfg(b)))

    def in_goal(self, p: np.ndarray) -> bool:
        return bool(self.sim.is_goal(to_cfg(p)))

    # ---- planning ---- #
    def plan(self, max_iters: int = 20000) -> Optional[List[np.ndarray]]:
        # Quick exits
        if self.in_goal(self.start):
            return [self.start.copy()]

        for it in range(1, max_iters + 1):
            q_rand = self.sample()
            idx = self.nearest_idx(q_rand)
            q_near = self.nodes[idx].p
            q_new = self.steer(q_near, q_rand)

            if not self.state_valid(q_new):
                continue
            if not self.edge_valid(q_near, q_new):
                continue

            new_idx = len(self.nodes)
            self.nodes.append(Node(q_new.copy(), parent=idx))

            # Goal test at the new node
            if self.in_goal(q_new):
                return self._reconstruct(new_idx)

            # Try to connect directly to goal center for faster success
            if self.edge_valid(q_new, self.goal_center):
                goal_idx = len(self.nodes)
                self.nodes.append(Node(self.goal_center.copy(), parent=new_idx))
                return self._reconstruct(goal_idx)

        return None

    def _reconstruct(self, idx: int) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        while idx is not None:
            node = self.nodes[idx]
            path.append(node.p.copy())
            idx = node.parent
        path.reverse()
        return path


# ------------------------- CLI ------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Single-drone RRT with obstacles (MultiDrone)")
    ap.add_argument("--env", type=str, default="single_drone_obs/env.yaml", help="Environment YAML file")
    ap.add_argument("--eta", type=float, default=1.0, help="Step size (meters) per extend")
    ap.add_argument("--iters", type=int, default=20000, help="Iteration")
    ap.add_argument("--goal-bias", type=float, default=0.1, help="Probability to sample the goal")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--no-viz", action="store_true", help="Skip visualization even if available")

    args = ap.parse_args()

    # Lazy import to avoid hard dependency at import-time
    from multi_drone import MultiDrone

    sim = MultiDrone(num_drones=1, environment_file=args.env)
    planner = RRT(sim, eta=args.eta, goal_bias=args.goal_bias, seed=args.seed)

    path = planner.plan(max_iters=args.iters)
    if path is None:
        print("No path found within the max iterations.")
        return
    print(f"Found path with {len(path)} waypoints.")

    # Convert to simulator shape for viz
    pos_traj = [p.reshape(1, 3) for p in path]

    if not args.no_viz and hasattr(sim, "visualize_paths"):
        try:
            sim.visualize_paths(pos_traj)
        except Exception as e:
            print(f"Visualization failed (continuing): {e}")

if __name__ == "__main__":
    main()
