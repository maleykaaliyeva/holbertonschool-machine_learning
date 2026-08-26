#!/usr/bin/env python3
"""
PCA module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a specified fraction of variance.

    Parameters:
        X (numpy.ndarray): Data matrix of shape (n, d) with zero mean.
        var (float): Fraction of original variance to maintain.

    Returns:
        W (numpy.ndarray): Weights matrix of shape (d, nd).
    """
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    cum_var = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.searchsorted(cum_var >= var, True) + 1
    W = Vt[:nd].T
    return W
