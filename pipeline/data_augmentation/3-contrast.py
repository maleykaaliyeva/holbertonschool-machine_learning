#!/usr/bin/env python3
"""Module for randomly adjusting image contrast using TensorFlow."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor containing the input image.
        lower: Float, lower bound for the contrast factor.
        upper: Float, upper bound for the contrast factor.

    Returns:
        A tf.Tensor representing the contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
