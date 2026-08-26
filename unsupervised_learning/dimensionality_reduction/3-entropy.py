#!/usr/bin/env python3
"""
Entropy module
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities relative to a data point.

    Parameters:
    - Di (numpy.ndarray): Pairwise distances of shape (n - 1,)
    - beta (numpy.ndarray): Beta value for Gaussian distribution

    Returns:
    - Hi (float): Shannon entropy of the points
    - Pi (numpy.ndarray): P affinities of the points of shape (n - 1,)
    """
    P = np.exp(-Di * beta)
    sum_P = np.sum(P)
    Pi = P / sum_P
    Hi = np.log2(sum_P) + beta * np.sum(Di * P) / (sum_P * np.log(2))

    return Hi, Pi
