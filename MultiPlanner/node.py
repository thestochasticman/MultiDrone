from typing_extensions import Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    q: np.ndarray         
    parent: Optional[int] 