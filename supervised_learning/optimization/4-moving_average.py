#!/usr/bin/env python3
"""Moving Average."""


def moving_average(data, beta):
    """Return a list containing the moving averages of data"""
    averages = []
    vt = 0
    for i in range(1, len(data) + 1):
        vt = beta * vt + (1 - beta) * data[i - 1]
        v_corrected = vt / (1 - beta ** i)
        averages.append(v_corrected)
    return averages
