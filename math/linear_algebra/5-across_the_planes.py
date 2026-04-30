#!/usr/bin/env python3
"""
Defines a function that adds two 2D matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices and returns a new matrix
    Returns None if matrices are not the same shape
    """
    # Check rows (number of lists inside the main list)
    if len(mat1) != len(mat2):
        return None

    # Check columns (number of items inside the first list)
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Create the new matrix using i (rows) and j (columns)
    new_matrix = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                  for i in range(len(mat1))]

    return new_matrix
