#!/usr/bin/env python3
"""Module that calculates the M-step in EM algorithm."""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    - X: numpy.ndarray of shape (n, d)
    - g: numpy.ndarray of shape (k, n)
    Returns: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    k, n = g.shape
    n_x, d = X.shape

    if n != n_x:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # pi: updated priors (average posterior probabilities)
    pi = np.sum(g, axis=1) / n

    # m: updated means
    # g shape: (k, n), X shape: (n, d)
    # m shape: (k, d)
    sum_g = np.sum(g, axis=1, keepdims=True)
    m = (g @ X) / sum_g

    # S: updated covariance matrices
    # S shape: (k, d, d)
    S = np.zeros((k, d, d))
    for i in range(k):
        # diff shape: (n, d)
        diff = X - m[i]
        # weighted outer product summed over n points, divided by sum of g_i
        S[i] = (g[i][:, np.newaxis] * diff).T @ diff / sum_g[i]

    return pi, m, S
