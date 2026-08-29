#!/usr/bin/env python3
"""
RNN Encoder module for machine translation
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder class using GRU
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab (int): size of the input vocabulary
            embedding (int): dimensionality of the embedding vectors
            units (int): number of hidden units in the GRU
            batch (int): batch size
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden state of the GRU to zeros

        Returns:
            tf.Tensor: initialized hidden state of shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass of the encoder

        Args:
            x (tf.Tensor): input tensor of shape (batch, input_seq_len)
            initial (tf.Tensor): initial hidden state (batch, units)

        Returns:
            outputs (tf.Tensor): GRU outputs (batch, input_seq_len, units)
            hidden (tf.Tensor): last hidden state (batch, units)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
