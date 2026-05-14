#!/usr/bin/env python3
"""
Module for Exponential distribution
"""


class Exponential:
    """
    Class that represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # lambtha = 1 / mean
            self.lambtha = float(1 / (sum(data) / len(data)))
