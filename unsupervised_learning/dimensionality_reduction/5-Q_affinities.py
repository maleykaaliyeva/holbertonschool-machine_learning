#!/usr/bin/env python3
"""
Q affinities module
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities in t-SNE.

    Parameters:
    - Y (numpy.ndarray): Low dimensional transformation of X (n, ndim)

    Returns:
    - Q (numpy.ndarray): Q affinities matrix of shape (n, n)
    - num (numpy.ndarray): Numerator matrix of Q affinities of shape (n, n)
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = sum_Y.reshape(-1, 1) + sum_Y.reshape(1, -1) - 2 * np.matmul(Y, Y.T)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)

    return Q, num
