#!/usr/bin/env python3
"""
Bayesian Information Criterion module
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds best number of clusters for GMM using BIC
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax or kmax > n:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if (not isinstance(tol, float) and not isinstance(tol, int)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    log_l_list = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    results = []

    for k in range(kmin, kmax + 1):
        try:
            res = expectation_maximization(
                X, k, iterations, tol, verbose
            )
            if res is None:
                return None, None, None, None
            pi, m, S, g, log_l = res
            if pi is None or m is None or S is None or log_l is None:
                return None, None, None, None
        except Exception:
            return None, None, None, None

        idx = k - kmin
        log_l_list[idx] = log_l
        results.append((pi, m, S))

        # Number of parameters p = k - 1 + k*d + k*d*(d+1)/2
        p = k - 1 + k * d + k * d * (d + 1) // 2
        b[idx] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_l_list, b
