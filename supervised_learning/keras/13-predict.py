#!/usr/bin/env python3
"""Prediction."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Generate predictions."""
    verbose = 1 if verbose else 0
    prediction = network.predict(data, verbose=verbose)
    return prediction
