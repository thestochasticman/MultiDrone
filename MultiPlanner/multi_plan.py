from MultiPlanner.utils import get_drone_sims
from Planner.multi_drone import MultiDrone
from typing_extensions import Optional
from Planner.utils import edge_valid
from Planner.samplers import sample
from Planner.rrt import try_connect
from Planner.utils import in_goal
from Planner.tree import Tree
import numpy as np
from copy import deepcopy
from Planner.plan import plan
import yaml


def multi_plan(
    sim: MultiDrone,
    config,
    max_iterations: int = 120000,
    eta: float = 0.8,
    goal_bias: float = 0.10,
    p_bridge: float = 0.40,
    p_obstacle: float = 0.40,
    ball_radius: float = 3.0,
    seed: Optional[int] = None,
    verbose: bool = True
):
    starts = np.asarray(sim.initial_configuration, float)
    goals = np.asarray(sim.goal_positions, float)

    single_drone_sims = get_drone_sims(starts, goals, config)

    paths = []
    for single_drone_sim in single_drone_sims:
        path = plan(
            single_drone_sim,
            max_iterations=max_iterations,
            eta=eta,
            goal_bias=goal_bias,
            p_bridge=p_bridge,
            p_obstacle=p_obstacle,
            ball_radius=ball_radius,
            seed=seed,
            verbose=True
        )
        paths += [path]

    
    current = starts.copy()
    multi_path = [current]
    while sum([len(p) for p in paths]) != 0:
        

    