#!/usr/bin/env python3
"""Module for calculating the adjugate matrix of a matrix."""


cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculate the adjugate matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The adjugate matrix of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.
    """
    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = [
        [cofactor_matrix[j][i] for j in range(len(cofactor_matrix))]
        for i in range(len(cofactor_matrix))
    ]

    return adjugate_matrix
