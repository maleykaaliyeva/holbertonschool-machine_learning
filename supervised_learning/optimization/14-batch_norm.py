#!/usr/bin/env python3
"""Batch Normalization Upgraded."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network."""
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg')
    )(prev)

    mean, variance = tf.nn.moments(dense, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    epsilon = 1e-7
    normalized = tf.nn.batch_normalization(
        dense, mean, variance, offset=beta, scale=gamma,
        variance_epsilon=epsilon
    )
    return activation(normalized)
