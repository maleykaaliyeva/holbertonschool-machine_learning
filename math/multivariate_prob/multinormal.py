#!/usr/bin/env python3
"""Initialize and PDF"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize the MultiNormal distribution."""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculate the PDF at a data point x."""
        d = self.mean.shape[0]

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        x_mean = x - self.mean

        coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        exponent = -0.5 * np.matmul(
            np.matmul(x_mean.T, inv),
            x_mean
        )

        return (coefficient * np.exp(exponent))[0][0]
