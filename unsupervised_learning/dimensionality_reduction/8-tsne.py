#!/usr/bin/env python3
"""
t-SNE module
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on a dataset.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d)
    - ndims (int): New dimensional representation
    - idims (int): Intermediate dimension after PCA
    - perplexity (float): Perplexity
    - iterations (int): Number of iterations
    - lr (float): Learning rate

    Returns:
    - Y (numpy.ndarray): Low dimensional transformation of shape (n, ndims)
    """
    X_pca = pca(X, idims)
    n, _ = X_pca.shape

    P = P_affinities(X_pca, perplexity=perplexity)
    P = P * 4
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    for step in range(1, iterations + 1):
        if step <= 20:
            alpha = 0.5
        else:
            alpha = 0.8

        dY, Q = grads(Y, P)
        iY = alpha * iY - lr * dY
        Y = Y + iY
        Y = Y - np.mean(Y, axis=0)

        if step % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(step, C))

        if step == 100:
            P = P / 4

    return Y
