#!/usr/bin/env python3
"""Neuron"""
import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron with gradient descent."""
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int) or step <= 0 or step > iterations:
                raise ValueError("step must be a positive intege
                                 r <= iterations")

        costs = []
        iteration_list = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                costs.append(c)
                iteration_list.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {c}")

        if graph:
            plt.plot(iteration_list, costs)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost over Iterations")
            plt.show()

        return self.evaluate(X, Y)
