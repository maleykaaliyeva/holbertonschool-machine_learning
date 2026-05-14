#!/usr/bin/env python3
"""
Module to calculate the likelihood of side effects using binomial distribution
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data x and n for each
    probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # nCr calculation: n! / (x!(n-x)!)
    # Using np.math.factorial for precision
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)

    combination = n_fact / (x_fact * nx_fact)

    # Likelihood = (nCr) * (P^x) * ((1-P)^(n-x))
    # Numpy will apply this to every element in the P array automatically
    return combination * (P ** x) * ((1 - P) ** (n - x))
