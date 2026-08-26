#!/usr/bin/env python3
"""
PCA v2 module
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Parameters:
    - X (numpy.ndarray): Data matrix of shape (n, d)
    - ndim (int): New dimensionality of transformed X

    Returns:
    - T (numpy.ndarray): Transformed version of X with shape (n, ndim)
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m, full_matrices=False)
    W = vh[:ndim].T
    T = np.matmul(X_m, W)
    return T
