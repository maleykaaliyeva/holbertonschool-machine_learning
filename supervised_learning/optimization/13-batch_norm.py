#!/usr/bin/env python3
"""Batch Normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Return the normalized Z matrix"""
    mean = np.mean(Z, axis=0)
    sigma = np.std(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(sigma ** 2 + epsilon)
    Z_final = gamma * Z_norm + beta
    return Z_final
