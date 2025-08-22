from MultiPlanner.multi_drone import MultiDrone
import numpy as np

def edge_valid(sim: MultiDrone, q1: np.ndarray, q2: np.ndarray):
    return sim.is_valid(q1, q2)

def clip_point(bounds: np.ndarray, q: np.ndarray)->np.ndarray:
  return np.clip(
    q,
    [
      bounds[0,0],
      bounds[1,0],
      bounds[2,0]
    ],
    [
      bounds[0,1],
      bounds[1,1],
      bounds[2,1]
    ]
    )

def is_state_valid(sim: MultiDrone, q: np.ndarray)->bool:
    return sim.is_valid(q)

def edge_valid(sim: MultiDrone, q1: np.ndarray, q2: np.ndarray)->bool:
    return sim.motion_valid(q1, q2)

def in_drone_goal(sim: MultiDrone, cfg: np.ndarray, drone_goal: np.ndarray)->bool:
    distances = np.linalg.norm(cfg - drone_goal, axis=1)
    return np.all(distances < sim._goal_radii)

