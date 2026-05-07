#!/usr/bin/env python3
"""Module for calculating the minor matrix of a matrix."""


determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculate the minor matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The minor matrix of matrix.

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

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for i in range(len(matrix)):
        minor_row = []

        for j in range(len(matrix)):
            sub_matrix = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix)
                if k != i
            ]

            minor_row.append(determinant(sub_matrix))

        minor_matrix.append(minor_row)

    return minor_matrix
