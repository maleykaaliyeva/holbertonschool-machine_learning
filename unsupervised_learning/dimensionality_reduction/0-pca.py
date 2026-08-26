#!/usr/bin/env python3
"""
PCA module
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    u, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    r = np.argwhere(cum_var >= var)[0][0]
    W = vh[:r + 1].T

    return W
