#!/usr/bin/env python3
"""Module that calculates intra-cluster variance."""
import numpy as np


def variance(X, C):
    """Calculates total intra-cluster variance."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    min_distances = np.min(distances, axis=1)
    return np.sum(min_distances)
