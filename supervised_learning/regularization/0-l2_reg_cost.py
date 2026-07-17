#!/usr/bin/env python3
"""L2 Regularization Cost."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Return cost of the model with L2 regularization term included."""
    # Cost is the cost with L2 regularization
    Cost = cost
    for i in range(1, L + 1):
        Cost += (lambtha / (2 * m)) * np.sum(np.square(weights[f"W{i}"]))

    return Cost
