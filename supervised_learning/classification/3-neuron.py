#!/usr/bin/env python3
"""Neuron."""
import numpy as np


class Neuron:
    """Neuron class."""

    def __init__(self, nx):
        """Constructor of the class."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weight getter."""
        return self.__W

    @property
    def b(self):
        """Bias getter."""
        return self.__b

    @property
    def A(self):
        """Activation getter."""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation."""
        self.__A = 1/(1+np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A

    def cost(self, Y, A):
        """Cost function."""
        return (-1 / len(Y[0])) * np.sum(Y * np.log(A) +
                                         (1 - Y) * np.log(1.0000001 - A))
