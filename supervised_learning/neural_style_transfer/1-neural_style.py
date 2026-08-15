#!/usr/bin/env python3
"""Module for Neural Style Transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Class that implements Neural Style Transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor for NST.

        Args:
            style_image (np.ndarray): Image used as a style reference.
            content_image (np.ndarray): Image used as a content reference.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.shape[-1] != 3 or len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           content_image.shape[-1] != 3 or len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels are between 0 and 1,

        with a maximum dimension of 512 pixels.
        """
        if not isinstance(image, np.ndarray) or \
           image.shape[-1] != 3 or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        image = tf.cast(image, tf.float32)
        max_dim = 512
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, new_shape)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        return image

    def load_model(self):
        """Creates the model used to calculate cost using VGG19 as base."""
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        return tf.keras.models.Model(
            inputs=vgg.input,
            outputs=model_outputs
        )
