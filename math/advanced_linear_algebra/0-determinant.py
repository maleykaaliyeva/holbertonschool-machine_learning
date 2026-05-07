#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The determinant of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) -
                (matrix[0][1] * matrix[1][0]))

    det = 0

    for col in range(len(matrix)):
        sub_matrix = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub_matrix)

    return det
