#!/usr/bin/env python3
"""Train."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """Return the History object generated after training the model."""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, shuffle=shuffle, verbose=verbose)
    return history
