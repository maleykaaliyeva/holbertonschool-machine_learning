#!/usr/bin/env python3
"""Module to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    res = [C]
    for i, coeff in enumerate(poly):
        if not isinstance(coeff, (int, float)):
            return None
        
        val = coeff / (i + 1)
        if val % 1 == 0:
            val = int(val)
        res.append(val)

    while len(res) > 1 and res[-1] == 0:
        res.pop()

    return res
