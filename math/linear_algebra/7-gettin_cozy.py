#!/usr/bin/env python3
""" Simple 2D concatenation """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates matrices based on axis """
    
    if axis == 0:
        # Check if widths match
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Just glue the lists of rows together
        return mat1 + mat2

    if axis == 1:
        # Check if heights match
        if len(mat1) != len(mat2):
            return None
        # Glue each individual row side-by-side
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
