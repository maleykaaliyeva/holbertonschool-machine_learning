#!/usr/bin/env python3
"""Neural Style Transfer."""

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
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(content_image, np.ndarray) \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """Scale an image to a maximum dimension of 512 pixels."""
        if not isinstance(image, np.ndarray) \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        height, width, _ = image.shape

        if height > width:
            new_height = 512
            new_width = int(width * 512 / height)
        else:
            new_width = 512
            new_height = int(height * 512 / width)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )
        image = image / 255.0

        return image

    def load_model(self):
        """Load the VGG19 model used for neural style transfer."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        def clone_layer(layer):
            """Clone a VGG19 layer, replacing max pooling with average pooling."""
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
            return layer.__class__.from_config(layer.get_config())

        model = tf.keras.models.clone_model(
            vgg,
            clone_function=clone_layer
        )

        model.set_weights(vgg.get_weights())
        model.trainable = False

        outputs = [
            model.get_layer(layer_name).output
            for layer_name in self.style_layers
        ]
        outputs.append(model.get_layer(self.content_layer).output)

        return tf.keras.Model(
            inputs=model.input,
            outputs=outputs
        )
