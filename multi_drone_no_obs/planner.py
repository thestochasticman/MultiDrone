#!/usr/bin/env python3

from typing import Dict, List, Set, Tuple
import numpy as np


def will_pair_collide(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    r: float,
    margin: float = 0.0
) -> Tuple[bool, float, float]:

    v = (b - a) - (d - c)
    w = (a - c)
    vv = float(np.dot(v, v))
    if vv == 0.0:
        dmin = float(np.linalg.norm(w))
        return (dmin < 2.0 * (r + margin), 0.0, dmin)
    t_star = - float(np.dot(w, v)) / vv
    t_star = max(0.0, min(1.0, t_star))
    dmin = float(np.linalg.norm(w + t_star * v))
    return (dmin < 2.0 * (r + margin), t_star, dmin)


def conflict_graph(starts: np.ndarray, goals: np.ndarray, r: float, margin: float = 0.0) -> Dict[int, Set[int]]:
    K = starts.shape[0]
    adj: Dict[int, Set[int]] = {i: set() for i in range(K)}
    for i in range(K):
        for j in range(i + 1, K):
            collides, _, _ = will_pair_collide(starts[i], goals[i], starts[j], goals[j], r, margin)
            if collides:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def greedy_layers(adj: Dict[int, Set[int]]) -> List[List[int]]:
    order = sorted(adj.keys(), key=lambda i: -len(adj[i]))  # highest degree first
    color: Dict[int, int] = {}
    for u in order:
        forbidden = {color[v] for v in adj[u] if v in color}
        c = 0
        while c in forbidden:
            c += 1
        color[u] = c
    layers: Dict[int, List[int]] = {}
    for i, c in color.items():
        layers.setdefault(c, []).append(i)
    return [layers[k] for k in sorted(layers.keys())]

def execute_layers_no_interp(sim, starts, goals, layers):

    path = [starts.copy()]
    current = starts.copy()

    for layer in layers:
        target = current.copy()
        idx = np.array(layer, dtype=int)
        target[idx] = goals[idx]

        # Try moving the whole layer at once
        if sim.motion_valid(current, target):
            path.append(target.copy())
            current = target
            continue

        # Otherwise, move layer members one-by-one (full edges only)
        for i in layer:
            one = current.copy()
            one[i] = goals[i]
            if not sim.motion_valid(current, one):
                raise RuntimeError(
                    f"Drone {i} cannot move straight to its goal without conflict. "
                    "Increase safety margin in conflict graph?."
                )
            path.append(one.copy())
            current = one

    return path


def get_drone_radius(sim, default: float = 0.3) -> float:
    return float(getattr(sim, "drone_radius", default))

def main():
    import argparse
    from multi_drone import MultiDrone

    p = argparse.ArgumentParser(description="Layered straight-line planner for multiple drones (no obstacles)")
    p.add_argument("--env", type=str, default="multi_drone_no_obs/env2.yaml", help="Environment YAML file")
    p.add_argument("--K", type=int, default=None, help="Number of drones (defaults to YAML)")
    p.add_argument("--margin", type=float, default=0.0, help="Extra safety margin added to radius in conflict test")

    args = p.parse_args()

    sim = MultiDrone(num_drones=args.K, environment_file=args.env)
    starts = np.asarray(sim.initial_configuration, dtype=float)
    goals  = np.asarray(sim.goal_positions, dtype=float)
    r = get_drone_radius(sim, default=0.3)

    if sim.motion_valid(starts, goals):
        path = [starts, goals]
    else:
        print('building conflict graph')

        adj = conflict_graph(starts, goals, r=r, margin=args.margin)
        layers = greedy_layers(adj)
        path = execute_layers_no_interp(sim, starts, goals, layers)


    print(f"Layers planned. Waypoints: {len(path)}")
    if hasattr(sim, "is_goal"):
        print("Goal reached?", sim.is_goal(path[-1]))

    sim.visualize_paths(path)


if __name__ == "__main__":
    main()
