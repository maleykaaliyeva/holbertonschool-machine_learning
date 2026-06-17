#!/usr/bin/env python3
"""Sensitivity."""
import numpy as np


def sensitivity(confusion):
    """Return sensitivity for each class."""
    n = confusion.shape[0]
    row_sums = np.sum(confusion, axis=1)

    return np.array([
        confusion[i][i] / row_sums[i] if row_sums[i] != 0 else 0
        for i in range(n)
    ])
