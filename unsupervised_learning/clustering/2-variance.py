#!/usr/bin/env python3
"""
Module to calculate intra-cluster variance.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    Parameters:
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - C is a numpy.ndarray of shape (k, d) containing the centroid means

    Returns:
    - The total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate squared distance between every point and every centroid
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)

    # Find the minimum distance (to the closest centroid) for each point
    min_distances = np.min(distances, axis=1)

    # Sum all the minimum distances to get total variance
    return np.sum(min_distances)
