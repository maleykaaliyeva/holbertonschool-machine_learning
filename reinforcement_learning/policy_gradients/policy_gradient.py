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
    if state.ndim == 1:
        state = state.reshape(1, -1)

    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Construct one-hot array Y for the action
    dsoftmax = np.zeros_like(probs)
    dsoftmax[0, action] = 1.0

    # Compute dlog = Y - P (Monte-Carlo policy gradient direction)
    dlog = dsoftmax - probs
    grad = np.dot(state.T, dlog)

    return action, grad
