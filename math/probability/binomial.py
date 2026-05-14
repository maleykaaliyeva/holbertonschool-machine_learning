#!/usr/bin/env python3
"""
Module for Binomial distribution
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)

            self.n = int(n_est)
            self.p = float(mean / n_est)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Factorial helper logic
        def factorial(n):
            res = 1
            for i in range(1, n + 1):
                res *= i
            return res

        n = self.n
        p = self.p

        # nCr = n! / (k! * (n-k)!)
        n_cr = factorial(n) / (factorial(k) * factorial(n - k))
        pmf_val = n_cr * (p ** k) * ((1 - p) ** (n - k))

        return pmf_val
