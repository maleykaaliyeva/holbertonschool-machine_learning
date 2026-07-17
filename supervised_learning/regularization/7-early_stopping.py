#!/usr/bin/env python3
"""Early Stopping"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Return a boolean of whether the network should be
    stopped early, followed by the updated count"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
