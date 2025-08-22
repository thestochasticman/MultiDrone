from Planner.utils import is_state_valid
from Planner.utils import edge_valid
from typing_extensions import Optional
from Planner.tree import Tree
import numpy as np


def steer(a: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
    d = float(np.linalg.norm(b - a))
    if d <= 1e-12:
        return a.copy()
    if d <= eta:
        return b.copy()
    return a + (eta / d) * (b - a)

def try_connect(sim, T: Tree, target: np.ndarray, eta: float) -> tuple[Optional[int], Optional[np.ndarray]]:
    """
    Extend T toward target; then greedily connect multiple steps (classic RRT-Connect).
    Returns (last_idx, last_q) if at least one step succeeded; else (None, None).
    """
    near_idx = T.nearest_idx(target)
    q_near = T.nodes[near_idx].q
    q_new = steer(q_near, target, eta)
    if not is_state_valid(sim, q_new) or not edge_valid(sim, q_near, q_new):
        return None, None

    last_idx = T.add(q_new, near_idx)
    last_q = q_new

    # greedy connect loop
    while True:
        q_step = steer(last_q, target, eta)

        if not is_state_valid(sim, q_step) or not edge_valid(sim, last_q, q_step):
            break
        last_idx = T.add(q_step, last_idx)
        last_q = q_step

        if float(np.linalg.norm(target - last_q)) < 0.25 * eta:
            break

    return last_idx, last_q
