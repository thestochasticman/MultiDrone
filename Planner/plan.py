from typing_extensions import Optional
from Planner.utils import fetch_bounds
from Planner.utils import edge_valid
from Planner.samplers import sample
from Planner.rrt import try_connect
from Planner.utils import in_goal
from Planner.tree import Tree
import numpy as np
import argparse

def plan(
    sim,
    max_iterations: int = 120000,
    eta: float = 0.8,
    goal_bias: float = 0.10,
    p_bridge: float = 0.40,
    p_obstacle: float = 0.40,
    ball_radius: float = 3.0,
    seed: Optional[int] = None,
    verbose: bool = True
)->list[np.ndarray]:
    
    rng = np.random.default_rng(seed)
    bounds = sim._bounds

    start = np.asarray(sim.initial_configuration, float).reshape(-1,3)[0]
    goal  = np.asarray(sim.goal_positions, float).reshape(-1,3)[0]

    # Quick path
    if edge_valid(sim, start, goal):
        return [start.copy(), goal.copy()]

    Ta = Tree(start)
    Tb = Tree(goal)

    for k in range(1, max_iterations + 1):

        q_rand = sample(sim, bounds, goal.copy(), goal_bias, p_bridge, p_obstacle, ball_radius, rng)

        idx_a, last_a = try_connect(sim, Ta, q_rand, eta)
        if idx_a is None:
            Ta, Tb = Tb, Ta
            continue

        idx_b, last_b = try_connect(sim, Tb, last_a, eta)
        if idx_b is not None and float(np.linalg.norm(last_a - last_b)) < 0.25 * eta:

            path_a = Ta.path_to_root(idx_a)
            path_b = Tb.path_to_root(idx_b)
            left = path_a
            right = list(reversed(path_b))
            if np.allclose(left[-1], right[0], atol=1e-9):
                right = right[1:]
            joint = left + right


            if not in_goal(sim, joint[-1]) and edge_valid(sim, joint[-1], goal):
                joint.append(goal.copy())

            if True:
                print(f" success at iter {k} with {len(joint)} waypoints.")
            return joint

        Ta, Tb = Tb, Ta

    if verbose:
        print("[RRT] failed: iteration budget exhausted.")
    return None
