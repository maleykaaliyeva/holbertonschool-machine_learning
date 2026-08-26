#!/usr/bin/env python3
"""
Initialize t-SNE module
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables required to calculate P affinities in t-SNE.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d)
    - perplexity (float): Perplexity for Gaussian distributions

    Returns:
    - D (numpy.ndarray): Squared pairwise distance matrix (n, n)
    - P (numpy.ndarray): P affinities matrix (n, n)
    - betas (numpy.ndarray): Beta values (n, 1)
    - H (float): Shannon entropy for perplexity with base 2
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
