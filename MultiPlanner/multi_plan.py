from MultiPlanner.multi_drone import MultiDrone
from typing_extensions import Optional
from MultiPlanner.plan import plan
import numpy as np

def multi_plan(
    sim: MultiDrone,
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

    start = np.asarray(sim.initial_configuration)
    goal = np.asarray(sim.goal_positions, float)

    cfg = start.copy()

    paths_per_drone = [False] * len(start)
    mega_path = []
    

    while not sim.is_goal(cfg):
        dists = ((start - goal)**2).sum(axis=1)
        print(dists.argsort())
        # drone_idx = dists.argmin()
        # if paths_per_drone[drone_idx] == False:
        #     drone_start = cfg.copy()
        #     drone_goal = cfg.copy()
        #     drone_goal[drone_idx] = goal[drone_idx]
        #     path = plan(
        #         sim,
        #         bounds,
        #         drone_start,
        #         drone_goal,
        #         drone_idx,
        #         max_iterations=max_iterations,
        #         eta=eta,
        #         goal_bias=goal_bias,
        #         p_bridge=p_bridge,
        #         p_obstacle=p_obstacle,
        #         ball_radius=ball_radius,
        #         seed=seed
        #     )
        #     paths_per_drone[drone_idx] = path
        #     cfg = paths_per_drone[drone_idx].pop(0)
        #     mega_path += [cfg]
        # else:
            

        #     cfg = paths_per_drone[drone_idx].pop(0)
        #     if sim.is_valid(cfg):
        #         mega_path += [cfg]
        #     else:
        #         path = plan(
        #         sim,
        #         bounds,
        #         drone_start,
        #         drone_goal,
        #         drone_idx,
        #         max_iterations=max_iterations,
        #         eta=eta,
        #         goal_bias=goal_bias,
        #         p_bridge=p_bridge,
        #         p_obstacle=p_obstacle,
        #         ball_radius=ball_radius,
        #         seed=seed
        #     )
            
        #     paths_per_drone[drone_idx] = path
        #     cfg = paths_per_drone[drone_idx].pop(0)
        #     mega_path += [cfg]

        
        
        # # print(mega_path[-1])
        
        