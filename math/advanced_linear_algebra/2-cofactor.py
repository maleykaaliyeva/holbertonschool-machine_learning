#!/usr/bin/env python3
"""Module for calculating the cofactor matrix of a matrix."""


minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculate the cofactor matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The cofactor matrix of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for i in range(len(minor_matrix)):
        cofactor_row = []

        for j in range(len(minor_matrix)):
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
