#!/usr/bin/env python3
"""Module that tests for the optimum number of clusters by variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of clusters
    - kmax: positive integer containing the maximum number of clusters
    - iterations: positive integer containing max iterations for K-means
    Returns: results, d_vars, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n

    if kmin >= kmax or kmin > n or kmax > n:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None

        if k == kmin:
            base_var = var
            d_vars.append(0.0)
        else:
            d_vars.append(base_var - var)

    return results, d_vars
