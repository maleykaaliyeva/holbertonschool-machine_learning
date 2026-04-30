#!/usr/bin/env python3
"""
Defines a function that transposes a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    """
    # Create a new matrix where we iterate through the columns of the original
    # For every column index 'i', we grab the element at that index from every row
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
