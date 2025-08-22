from MultiPlanner.node import Node
from typing import List
import numpy as np

class Tree:
    def __init__(self, root: np.ndarray):
        self.nodes: List[Node] = [Node(root.copy(), parent=None)]

    def add(self, q: np.ndarray, parent_idx: int) -> int:
        self.nodes.append(Node(q.copy(), parent=parent_idx))
        return len(self.nodes) - 1

    def nearest_idx(self, q: np.ndarray, drone_idx: int) -> int:
        drone_q = q[drone_idx].reshape(3)
        P = np.asarray([n.q[drone_idx].ravel() for n in self.nodes])
        d2 = ((P - drone_q)**2).sum(axis=1)
        return int(np.argmin(d2))

    def path_to_root(self, idx: int) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        while idx is not None:
            out.append(self.nodes[idx].q.copy())
            idx = self.nodes[idx].parent
        out.reverse()
        return out