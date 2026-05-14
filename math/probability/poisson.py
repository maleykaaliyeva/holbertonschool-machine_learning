#!/usr/bin/env python3
"""
Contains the Poisson class that represents a poisson distribution
"""


class Poisson:
    """ Class Poisson that represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the Poisson distribution
        """
        if data is None:
            # Check lambtha only if data is not provided
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validation for the provided data list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate lambtha as the mean of the data
            self.lambtha = float(sum(data) / len(data))
