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
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
