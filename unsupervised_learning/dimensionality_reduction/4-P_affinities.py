#!/usr/bin/env python3
"""
P affinities module
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d)
    - tol (float): Maximum tolerance allowed for entropy difference
    - perplexity (float): Target perplexity

    Returns:
    - P (numpy.ndarray): Symmetric P affinities of shape (n, n)
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        betamin = None
        betamax = None
        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, betas[i])
        H_diff = Hi - H

        while np.abs(H_diff) > tol:
            if H_diff > 0:
                betamin = betas[i, 0]
                if betamax is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamax) / 2
            else:
                betamax = betas[i, 0]
                if betamin is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamin) / 2

            Hi, Pi = HP(Di, betas[i])
            H_diff = Hi - H

        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    P = (P + P.T) / (2 * n)

    return P
