#!/usr/bin/env python3
"""Defines the Neural Style Transfer class."""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the NST instance."""
        if not isinstance(style_image, np.ndarray) \
                or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Scale an image to a maximum dimension of 512 pixels."""
        if not isinstance(image, np.ndarray) \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        scale = 512 / tf.cast(tf.maximum(height, width), tf.float32)

        new_height = tf.cast(
            tf.cast(height, tf.float32) * scale,
            tf.int32
        )
        new_width = tf.cast(
            tf.cast(width, tf.float32) * scale,
            tf.int32
        )

        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )

        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)

        return tf.expand_dims(image, axis=0)

    def load_model(self):
        """Load VGG19 and create the NST model."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        def replace_pooling(layer):
            """Replace max pooling layers with average pooling layers."""
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
            return layer

        model = tf.keras.models.clone_model(
            vgg,
            clone_function=replace_pooling
        )

        model.set_weights(vgg.get_weights())
        model.trainable = False

        outputs = [
            model.get_layer(layer_name).output
            for layer_name in self.style_layers
        ]
        outputs.append(model.get_layer(self.content_layer).output)

        self.model = tf.keras.Model(
            inputs=model.input,
            outputs=outputs
        )

        self.model.trainable = False
