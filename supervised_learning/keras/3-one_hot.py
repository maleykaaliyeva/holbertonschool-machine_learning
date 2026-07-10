#!/usr/bin/env python3
"""One-hot encoding"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Return one-hot matrix."""
    one_hot_matrix = K.utils.to_categorical(labels, classes)
    return one_hot_matrix
