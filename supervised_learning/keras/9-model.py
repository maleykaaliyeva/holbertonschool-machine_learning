#!/usr/bin/env python3
"""Save and Load."""
import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire model."""
    network.save(filename)


def load_model(filename):
    """Load an entire model."""
    model = K.models.load_model(filename)
    return model
