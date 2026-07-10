#!/usr/bin/env python3
"""Momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Return the updated variable and the new moment, respectively"""
    v = v * beta1 + (1 - beta1) * grad
    var = var - v * alpha
    return var, v
