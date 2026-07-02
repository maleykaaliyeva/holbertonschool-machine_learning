#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class NeuralNetwork:
    """Class"""

    def __init__(self, nx, nodes):
        """Constructor of the class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weight getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Bias getter for b1."""
        return self.__b1

    @property
    def A1(self):
        """Activation getter for A1."""
        return self.__A1

    @property
    def W2(self):
        """Weight getter for W2."""
        return self.__W2

    @property
    def b2(self):
        """Bias getter for b2."""
        return self.__b2

    @property
    def A2(self):
        """Activation getter for A2."""
        return self.__A2

    def forward_prop(self, X):
        """Forward propagation."""
        self.__A1 = 1/(1 + np.exp(-np.dot(self.__W1, X) - self.__b1))
        self.__A2 = 1/(1 + np.exp(-np.dot(self.__W2, self.__A1) - self.__b2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Cost function"""
        c = -1 / len(Y[0]) * np.sum((Y * np.log(A)) +
                                    (1 - Y) * np.log(1.0000001 - A))
        return c
