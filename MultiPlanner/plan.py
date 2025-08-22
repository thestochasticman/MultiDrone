from MultiPlanner.multi_drone import MultiDrone
from MultiPlanner.utils import in_drone_goal
from MultiPlanner.samplers import sample
from MultiPlanner.rrt import try_connect
from MultiPlanner.samplers import sample
from MultiPlanner.utils import edge_valid
from MultiPlanner.run_info import RunInfo
from typing_extensions import Optional
from MultiPlanner.tree import Tree
import time
import numpy as np

def plan(
    sim: MultiDrone,
    bounds: np.ndarray,
    drone_start: np.ndarray,
    drone_goal: np.ndarray,
    drone_idx: np.ndarray,
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

    Ta = Tree(drone_start)
    Tb = Tree(drone_goal)

    start_is_Ta = True
    cfg = drone_start.copy()

    start_time = time.perf_counter()
    for k in range(1, max_iterations + 1):
        q_rand = sample(
            sim,
            bounds,
            cfg,
            drone_idx=drone_idx,
            goal=drone_goal,
            goal_bias=goal_bias,
            p_bridge=p_bridge,
            p_obstacle=p_obstacle,
            ball_radius=ball_radius,
            rng=rng
        )

        idx_a, last_a = try_connect(sim, drone_idx, Ta, q_rand, eta)
        
        if idx_a is None:
            Ta, Tb = Tb, Ta
            start_is_Ta = not start_is_Ta
            continue

        idx_b, last_b = try_connect(sim, drone_idx, Tb, last_a, eta)
        if idx_b is not None and float(np.linalg.norm(last_a[drone_idx] - last_b[drone_idx])) < 0.25 * eta:
            
       
            if start_is_Ta:
                path_start = Ta.path_to_root(idx_a)
                path_goal  = Tb.path_to_root(idx_b)   
            else:
                path_start = Tb.path_to_root(idx_b)   
                path_goal  = Ta.path_to_root(idx_a)   

            left  = path_start
            right = list(reversed(path_goal))        
            if np.allclose(left[-1], right[0], atol=1e-9):
                right = right[1:]
            joint = left + right

            if not in_drone_goal(sim, joint[-1], drone_goal) and edge_valid(sim, joint[-1], drone_goal):
                joint.append(drone_goal.copy())

            time_taken = time.perf_counter() - start_time

            run_info = RunInfo(int(drone_idx), k, len(joint), time_taken)
            if verbose:
                # print(f" success at iter {k} with {len(joint)} waypoints.")
                print(run_info)
            return joint, run_info
        
        Ta, Tb = Tb, Ta
        start_is_Ta = not start_is_Ta
