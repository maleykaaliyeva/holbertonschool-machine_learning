#!/usr/bin/env python3
"""
Self-attention mechanism for machine translation
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-attention layer based on Bahdanau attention
    """

    def __init__(self, units):
        """
        Class constructor

        Args:
            units (int): number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Computes the context vector and attention weights

        Args:
            s_prev (tf.Tensor): previous decoder hidden state
                                shape (batch, units)
            hidden_states (tf.Tensor): encoder outputs
                                       shape (batch, input_seq_len, units)

        Returns:
            context (tf.Tensor): context vector for decoder
                                 shape (batch, units)
            weights (tf.Tensor): attention weights
                                 shape (batch, input_seq_len, 1)
        """
        # Expand decoder hidden state to match time dimension
        s_prev = tf.expand_dims(s_prev, axis=1)

        # Alignment scores
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev) + self.U(hidden_states)
            )
        )

        # Attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Context vector as weighted sum of encoder outputs
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
