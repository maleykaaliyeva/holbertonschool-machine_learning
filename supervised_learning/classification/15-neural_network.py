#!/usr/bin/env python3
"""Neural Network"""
import numpy as np
import matplotlib.pyplot as plt


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
        c = -1 / len(Y[0]) * np.sum((Y * np.log(A)) + (1 - Y)
                                    * np.log(1.0000001 - A))
        return c

    def evaluate(self, X, Y):
        """Evaluate the model"""
        self.forward_prop(X)
        return np.where(self.__A2 >= 0.5, 1, 0), self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Gradient descent"""
        m = len(Y[0])

        dZ2 = A2 - Y
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1/m) * np.matmul(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Training."""
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int) or step <= 0 or step > iterations:
                raise ValueError("step must be a positive integer\
                                 <= iterations")

        costs = []
        iteration_list = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

            if i % step == 0 or i == iterations:
                c = self.cost(Y, A2)
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
        self.forward_prop(X)
        return self.evaluate(X, Y)
