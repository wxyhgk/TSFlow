from __future__ import annotations

import numpy as np


def max_force(forces) -> float:
    vectors = np.asarray(forces, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(vectors, axis=1).max(initial=0.0))
