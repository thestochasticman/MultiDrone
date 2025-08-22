from MultiPlanner.utils import is_state_valid
from typing_extensions import Optional
from MultiPlanner.utils import clip_point
from MultiPlanner.multi_drone import MultiDrone
import numpy as np



def sample_uniform(
    sim: MultiDrone,
    bounds: np.ndarray,
    cfg: np.ndarray,
    drone_idx: int,
    rng: np.random.Generator
) -> np.ndarray:

    sampled_pos_drone = np.array(
        [
            rng.uniform(bounds[0,0], bounds[0,1]),
            rng.uniform(bounds[1,0], bounds[1,1]),
            rng.uniform(bounds[2,0], bounds[2,1])
        ]
    )
    new_cfg = cfg.copy()
    new_cfg[drone_idx] = sampled_pos_drone

    while not is_state_valid(sim, new_cfg):
        sampled_pos_drone = np.array(
            [
                rng.uniform(bounds[0,0], bounds[0,1]),
                rng.uniform(bounds[1,0], bounds[1,1]),
                rng.uniform(bounds[2,0], bounds[2,1])
            ]
        )
        new_cfg = new_cfg.copy()
        new_cfg[drone_idx] = sampled_pos_drone
    return new_cfg

def sample_inside_ball(
        drone_idx: int,
        bounds: np.ndarray,
        q1: np.ndarray,
        ball_radius: int=3,
        rng=np.random.Generator
    ) -> np.ndarray:
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        u = np.random.choice(
            np.array[[1.0, 0.0, 0.0]],
            np.array[[0.0, 1.0, 0.0]],
            np.array[[0.0, 0.0, 0.1]],
        )
    else:
        u = v / norm   
    r = ball_radius * (rng.random() ** (1/3))
    q2 = q1.copy()
    q2[drone_idx] = q1[drone_idx] + ball_radius * u
    return clip_point(bounds, q2)

def sample_near_obstacle(
        sim,
        bounds: np.ndarray,
        cfg: np.ndarray,
        drone_idx: np.ndarray,
        rng: np.random.Generator,
        max_iter: int = 20,
        ball_radius=3
    )-> np.ndarray:

    for iteration in range(max_iter):
        q1 = sample_uniform(sim, bounds, cfg, drone_idx, rng)
        q2 = sample_inside_ball(drone_idx, bounds, q1, ball_radius, rng)
        q1_valid = is_state_valid(sim, q1)
        q2_valid = is_state_valid(sim, q2)

        if q1_valid and not q2_valid:
            return q1
        if not q1_valid and q2_valid:
            return q2

    return sample_uniform(sim, bounds, cfg, drone_idx, rng)

def sample_bridge(
    sim:np.ndarray,
    bounds: np.ndarray,
    cfg: np.ndarray,
    drone_idx: int,
    rng: np.random.Generator,
    max_iterations: int = 30
)->Optional[np.ndarray]:
    
    for iteration in range(max_iterations):
        q1 = sample_uniform(sim, bounds, cfg, drone_idx, rng)
        q2 = sample_uniform(sim, bounds, cfg, drone_idx, rng)

        valid_q1 = is_state_valid(sim, q1)
        valid_q2 = is_state_valid(sim, q2)

        if not valid_q1 and not valid_q2:
            m = 0.5 * (q1 + q2)
            if is_state_valid(sim, m):
                return m
    return sample_uniform(sim, bounds, cfg, drone_idx, rng)

def sample(
    sim: MultiDrone,
    bounds: np.ndarray,
    cfg: np.ndarray,
    drone_idx: int, 
    goal: np.ndarray,
    goal_bias: float,
    p_bridge: float,
    p_obstacle: float,
    ball_radius: float,
    rng: np.random.Generator,
)->np.ndarray:
    

    r = rng.random()
    if r < goal_bias:
        return goal.copy()
    else:
        r2 = (r - goal_bias) / max(1e-12, 1 - goal_bias)
        if r2 < p_bridge:
            return sample_bridge(sim, bounds, cfg, drone_idx, rng)
        elif r2 < p_bridge + p_obstacle:
            return sample_near_obstacle(sim, bounds, cfg, drone_idx, rng, ball_radius=ball_radius)
        else:
            return sample_uniform(sim, bounds, cfg, drone_idx, rng)

