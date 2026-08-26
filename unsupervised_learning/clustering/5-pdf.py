#!/usr/bin/env python3
"""Module that calculates the PDF of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a Gaussian distribution.
    - X: numpy.ndarray of shape (n, d)
    - m: numpy.ndarray of shape (d,)
    - S: numpy.ndarray of shape (d, d)
    Returns: P, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0] or \
       S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    # S_inv: Inverse covariance matrix
    # S_det: Determinant of covariance matrix
    S_inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)

    # Term 1: 1 / sqrt((2 * pi)^d * det(S))
    coef = 1 / np.sqrt(((2 * np.pi) ** d) * S_det)

    # Term 2: exp(-0.5 * (x - m)^T * S_inv * (x - m))
    # Calculate (x - m)
    diff = X - m

    # Calculate (diff * S_inv) * diff, summed across axis 1
    # This efficiently performs the quadratic form without loops
    exponent = -0.5 * np.sum((diff @ S_inv) * diff, axis=1)

    P = coef * np.exp(exponent)

    # All values in P should have a minimum value of 1e-300
    P = np.maximum(P, 1e-300)

    return P
