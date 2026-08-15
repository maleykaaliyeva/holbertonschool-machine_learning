#!/usr/bin/env python3
"""
Neural Style Transfer module with Feature Extraction
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs tasks for neural style transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor for NST class.

        Parameters:
            style_image (np.ndarray): Image used as a style reference.
            content_image (np.ndarray): Image used as a content reference.
            alpha (float/int): Weight for content cost.
            beta (float/int): Weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Parameters:
            image (np.ndarray): Image to be scaled of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image of shape (1, h_new, w_new, 3).
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        scaled = tf.image.resize(
            tf.expand_dims(image, axis=0),
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        scaled = scaled / 255.0
        scaled = tf.clip_by_value(scaled, 0.0, 1.0)

        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as a base.
        Replaces MaxPooling2D layers with AveragePooling2D layers.
        Sets model outputs to style layers followed by content layer.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs_dict = {}

        x = vgg.input
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    name=layer.name
                )
            layer.trainable = False
            x = layer(x)
            if (layer.name in self.style_layers or
                    layer.name == self.content_layer):
                outputs_dict[layer.name] = x

        style_outputs = [outputs_dict[layer] for layer in self.style_layers]
        content_output = outputs_dict[self.content_layer]

        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer tensor.

        Parameters:
            input_layer (tf.Tensor|tf.Variable): Tensor of shape (1, h, w, c).

        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = input_layer.shape[-1]
        image_shape = tf.shape(input_layer)
        h = tf.cast(image_shape[1], tf.float32)
        w = tf.cast(image_shape[2], tf.float32)

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return gram / (h * w)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.

        Sets instance attributes:
            gram_style_features: list of gram matrices for style layer outputs
            content_feature: content layer output of content image
        """
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_preprocessed)
        content_outputs = self.model(content_preprocessed)

        self.gram_style_features = [
            self.gram_matrix(layer)
            for layer in style_outputs[:len(self.style_layers)]
        ]
        self.content_feature = content_outputs[-1]
