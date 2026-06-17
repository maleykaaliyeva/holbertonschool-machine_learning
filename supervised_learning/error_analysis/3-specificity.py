#!/usr/bin/env python3
"""Specificity."""
import numpy as np


def specificity(confusion):
    """Return a numpy.ndarray of shape (classes,) containing
    the specificity of each class."""
    n = confusion.shape[0]
    spec_mat = np.zeros((n,))
    for i in range(n):
        TN = (np.sum(confusion) - np.sum(confusion[i, :])
              - np.sum(confusion[:, i]) + confusion[i][i])
        FP = np.sum(confusion[:, i]) - confusion[i][i]
        spec_mat[i] = (TN / (TN + FP)) if TN + FP != 0 else 0
    return spec_mat
