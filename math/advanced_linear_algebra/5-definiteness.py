#!/usr/bin/env python3
"""Module for calculating the definiteness of a matrix."""


import numpy as np


def definiteness(matrix):
    """Calculate the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n).

    Returns:
        The definiteness of the matrix, or None if invalid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except Exception:
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    return "Indefinite"
