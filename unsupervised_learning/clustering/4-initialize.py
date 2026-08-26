#!/usr/bin/env python3
"""Module that initializes variables for a Gaussian Mixture Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a GMM.
    - X: numpy.ndarray of shape (n, d)
    - k: positive integer (number of clusters)
    Returns: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # pi: initialized evenly
    pi = np.full((k,), 1 / k)

    # m: initialized with K-means
    m, _ = kmeans(X, k)

    # S: covariance matrices initialized as identity matrices
    # Shape should be (k, d, d)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
