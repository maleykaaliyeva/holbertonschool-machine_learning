#!/usr/bin/env python3
"""Normalization Constants."""
import numpy as np


def normalization_constants(X):
    """Return normalization constants."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
