#!/usr/bin/env python3
"""Testing model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test model."""
    verbose = 1 if verbose else 0
    result = network.evaluate(data, labels, verbose=verbose)
    return result
