#!/usr/bin/env python3
"""RMSProp Upgraded"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Return optimizer."""
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)
    return optimizer
