#!/usr/bin/env python3
"""
Policy Gradient functions
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes policy probabilities using softmax

    Args:
        matrix: state matrix
        weight: weight matrix

    Returns:
        softmax probability matrix
    """
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient

    Args:
        state: matrix representing current state
        weight: matrix of weights

    Returns:
        action, gradient
    """
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    state = np.array(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)

    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1.0

    grad = np.dot(state.T, dsoftmax)

    return action, grad
