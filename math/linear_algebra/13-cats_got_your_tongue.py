#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis using numpy
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Returns a new numpy.ndarray that is the concatenation of mat1 and mat2
    """
    return np.concatenate((mat1, mat2), axis=axis)
