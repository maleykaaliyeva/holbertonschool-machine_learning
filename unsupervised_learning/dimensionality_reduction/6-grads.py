#!/usr/bin/env python3
"""
Gradients module
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y in t-SNE.

    Parameters:
    - Y (numpy.ndarray): Low dimensional transformation of X of shape (n, ndim)
    - P (numpy.ndarray): P affinities matrix of shape (n, n)

    Returns:
    - dY (numpy.ndarray): Gradients of Y of shape (n, ndim)
    - Q (numpy.ndarray): Q affinities of Y of shape (n, n)
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    PQ_diff = (P - Q) * num
    dY = np.zeros((n, ndim))

    for i in range(n):
        dY[i] = np.sum((PQ_diff[i, :, None]) * (Y[i] - Y), axis=0)

    return dY, Q
