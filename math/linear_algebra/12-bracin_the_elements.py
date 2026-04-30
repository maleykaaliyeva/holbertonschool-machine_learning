#!/usr/bin/env python3
"""
Performs element-wise operations using numpy
"""


def np_elementwise(mat1, mat2):
    """
    Returns a tuple containing the sum, difference, product, and quotient
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return (add, sub, mul, div)
