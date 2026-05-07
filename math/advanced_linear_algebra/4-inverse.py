#!/usr/bin/env python3
"""Module for calculating the inverse of a matrix."""


determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Calculate the inverse of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The inverse of matrix, or None if matrix is singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    adj = adjugate(matrix)

    inverse_matrix = [
        [element / det for element in row]
        for row in adj
    ]

    return inverse_matrix
