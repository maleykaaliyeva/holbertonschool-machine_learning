#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update the weights and biases of a neural network
    using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # output layer error (softmax derivative)

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]

        # Add L2 term to gradient
        dW = (dZ @ A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Compute dZ_prev before updating weights
        if i > 1:
            dZ_prev = (W.T @ dZ) * (1 - A_prev ** 2)  # derivative of tanh

        # Update weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            dZ = dZ_prev
