#!/usr/bin/env python3
"""
Neural Style Transfer module (Task 10)
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs tasks for neural style transfer
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1,
                 var=10):
        """
        Constructor for NST class.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )
        if (not isinstance(alpha, (int, float, np.number)) or
                isinstance(alpha, bool) or alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )
        if (not isinstance(beta, (int, float, np.number)) or
                isinstance(beta, bool) or beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )
        if (not isinstance(var, (int, float, np.number)) or
                isinstance(var, bool) or var < 0):
            raise TypeError(
                "var must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1.
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
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
        Creates the model used to calculate cost from VGG19 Keras base
        model, replacing MaxPooling2D layers with AveragePooling2D.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.save("vgg_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "vgg_base_model", custom_objects=custom_objects
        )

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        """
        vgg19 = tf.keras.applications.vgg19

        preprocess_style = vgg19.preprocess_input(self.style_image * 255)
        preprocess_content = vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(preprocess_style)[:-1]
        content_output = self.model(preprocess_content)[-1]

        self.gram_style_features = [
            self.gram_matrix(style_layer) for style_layer in style_outputs
        ]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = int(style_output.shape[-1])
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c
                )
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.
        """
        length = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != length):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1 / length
        style_cost = 0

        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):
            style_cost += weight * self.layer_style_cost(
                style_output, gram_target
            )

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image.
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )

        content_cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

        return content_cost

    @staticmethod
    def variational_cost(generated_image):
        """
        Calculates the variational cost for the generated image.
        """
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) not in (3, 4)):
            raise TypeError(
                "image must be a tensor of rank 3 or 4"
            )

        return tf.reduce_sum(tf.image.total_variation(generated_image))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image.
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        vgg19 = tf.keras.applications.vgg19
        preprocess_generated = vgg19.preprocess_input(generated_image * 255)

        outputs = self.model(preprocess_generated)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)
        J = self.alpha * J_content + self.beta * J_style + self.var * J_var

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image.
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s)
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image
            )

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        Generates the neural style transferred image.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            gradients, J_total, J_content, J_style, J_var = \
                self.compute_grads(generated_image)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                msg = ("Cost at iteration {}: {}, content {}, "
                       "style {}, var {}")
                print(msg.format(
                    i, J_total, J_content, J_style, J_var
                ))

            if i < iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        generated_image = best_image[0]

        return generated_image, best_cost
