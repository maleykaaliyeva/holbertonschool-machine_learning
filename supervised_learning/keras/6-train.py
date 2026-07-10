#!/usr/bin/env python3
"""Early Stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Return the History object generated after training the model."""
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience))
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callbacks, shuffle=shuffle,
                          verbose=verbose)
    return history
