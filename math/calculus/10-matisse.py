#!/usr/bin/env python3
"""Module to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    # Derivative rule: power * coefficient, index is the power
    derivative = [poly[i] * i for i in range(1, len(poly))]

    # Clean up trailing zeros to match Holberton's expected output
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
