#!/usr/bin/env python3
""" Learning Rate Decay Upgraded."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Return the updated value for alpha."""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
