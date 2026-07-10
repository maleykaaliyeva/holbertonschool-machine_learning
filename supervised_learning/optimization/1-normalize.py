#!/usr/bin/env python3
"""Normalization."""
import numpy as np


def normalize(X, m, s):
    """Return the normalized matrix."""
    X = (X - m) / s
    return X
