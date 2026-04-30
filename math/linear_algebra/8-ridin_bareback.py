#!/usr/bin/env python3
"""
Defines a function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication on two 2D matrices
    """
    # Check if multiplication is possible
    # Columns of mat1 must equal Rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Get dimensions
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    cols2 = len(mat2[0])

    # Create a new matrix filled with zeros
    # Shape will be (rows of mat1) x (columns of mat2)
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Perform multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
