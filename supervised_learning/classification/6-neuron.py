#!/usr/bin/env python3
"""Neuron"""
import numpy as np


class Neuron:
    """Neuron class"""

    def __init__(self, nx):
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
        """Weight getter"""
        return self.__W

    @property
    def b(self):
        """Bias getter"""
        return self.__b

    @property
    def A(self):
        """Activation getter"""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation"""
        self.__A = 1/(1+np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A

    def cost(self, Y, A):
        """Cost function"""
        return (-1 / len(Y[0])) * np.sum(Y * np.log(A) +
                                         (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluate the neuron"""
        self.forward_prop(X)
        return np.where(self.__A >= 0.5, 1, 0), self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient descent"""
        dZ = A - Y
        dW = (1/len(Y[0])) * np.dot(dZ, X.T)
        db = (1/len(Y[0])) * np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Training."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
