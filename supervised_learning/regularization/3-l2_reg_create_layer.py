#!/usr/bin/env python3
"""Creating a Layer with L2 Regularization."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create layer with L2"""
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer_weight = tf.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=layer_weight
    )(prev)
    return layer
