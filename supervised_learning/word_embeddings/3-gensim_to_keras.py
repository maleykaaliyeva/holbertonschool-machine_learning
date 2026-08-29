#!/usr/bin/env python3
"""
Gensim to Keras Embedding conversion module
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim word2vec model
    to a trainable keras Embedding layer.
    """
    weights = model.wv.vectors
    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return layer
