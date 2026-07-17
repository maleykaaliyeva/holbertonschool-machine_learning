#!/usr/bin/env python3
"""Gradient Descent with Dropout."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network with Dropout
    regularization using gradient descent.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # output layer error (softmax derivative)

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]

        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Backpropagate error if not input layer
        if i > 1:
            W = weights[f"W{i}"]
            dZ = W.T @ dZ
            A_prev_current = cache[f"A{i-1}"]
            D = cache[f"D{i-1}"]
            dZ = dZ * D / keep_prob
            dZ = dZ * (1 - A_prev_current ** 2)  # derivative of tanh

        # Update weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
