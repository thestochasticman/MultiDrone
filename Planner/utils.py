import numpy as np

def fetch_bounds(sim) -> np.ndarray:
		return sim._bounds

def cfg_from_point(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, float).reshape(1, 3)

def is_state_valid(sim, p: np.ndarray) -> bool:
    return sim.is_valid(cfg_from_point(p))

def edge_valid(sim, a: np.ndarray, b: np.ndarray) -> bool:
    return bool(sim.motion_valid(cfg_from_point(a), cfg_from_point(b)))

def in_goal(sim, p: np.ndarray) -> bool:
    return bool(sim.is_goal(cfg_from_point(p)))


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