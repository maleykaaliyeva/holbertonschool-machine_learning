#!/usr/bin/env python3
"""
Module for Normal distribution
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            sum_diff_sq = sum([(x - self.mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        mean = self.mean
        stddev = self.stddev

        exponent = -0.5 * ((x - mean) / stddev) ** 2
        coefficient = 1 / (stddev * ((2 * pi) ** 0.5))

        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        mean = self.mean
        stddev = self.stddev

        # y = (x - mean) / (stddev * sqrt(2))
        y = (x - mean) / (stddev * (2 ** 0.5))

        # erf(y) approximation
        erf = (2 / (pi ** 0.5)) * (y - (y**3 / 3) + (y**5 / 10) -
                                   (y**7 / 42) + (y**9 / 216))

        return 0.5 * (1 + erf)
