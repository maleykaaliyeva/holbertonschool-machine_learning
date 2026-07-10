#!/usr/bin/env python3
"""Validation data"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Return the History object generated after training the model."""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          shuffle=shuffle, verbose=verbose)
    return history
