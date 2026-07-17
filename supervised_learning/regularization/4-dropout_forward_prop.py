#!/usr/bin/env python3
"""Forward Propagation with Dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return a dictionary containing the outputs of each
    layer and the dropout mask used on each layer."""
    cache = {}
    cache["A0"] = X
    for i in range(1, L):
        W = weights["W{}".format(i)]
        A = cache["A{}".format(i-1)]
        b = weights["b{}".format(i)]
        z = np.dot(W, A) + b
        A = np.tanh(z)   # we compute A to match the shape with D
        D = (np.random.rand(*A.shape) < keep_prob).astype(int)
        A = A * D / keep_prob
        cache["A{}".format(i)] = A
        cache["D{}".format(i)] = D

    # last layer
    z = (np.dot(weights["W{}".format(L)], cache["A{}".format(L-1)]) +
         weights["b{}".format(L)])

    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    cache["A{}".format(L)] = A
    return cache
