#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Return the History object generated after training the model."""
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience))
    if learning_rate_decay and validation_data is not None:
        def decay(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(decay,
                                                           verbose=1))

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callbacks, shuffle=shuffle,
                          verbose=verbose)
    return history
