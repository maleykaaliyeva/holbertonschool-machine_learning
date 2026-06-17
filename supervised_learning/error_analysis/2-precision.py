#!/usr/bin/env python3
"""Precision."""
import numpy as np


def precision(confusion):
    """Returns precision for each class."""
    n = confusion.shape[0]
    col_sums = np.sum(confusion, axis=0)

    return np.array([
        confusion[i][i] / col_sums[i] if col_sums[i] != 0 else 0
        for i in range(n)
    ])
