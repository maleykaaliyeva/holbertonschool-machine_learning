#!/usr/bin/env python3
"""F1 score."""
import numpy as np


def f1_score(confusion):
    """Return a numpy.ndarray of shape (classes,) containing
    the F1 score of each class."""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    return (2 * sensitivity(confusion) * precision(confusion) /
            (sensitivity(confusion) + precision(confusion)))
