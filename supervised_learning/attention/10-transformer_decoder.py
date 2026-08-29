#!/usr/bin/env python3
"""
Transformer Decoder
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder
    """

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): number of decoder blocks
            dm (int): dimensionality of the model
            h (int): number of attention heads
            hidden (int): hidden units in the feed-forward network
            target_vocab (int): size of target vocabulary
            max_seq_len (int): maximum sequence length
            drop_rate (float): dropout rate
        """
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            target_vocab, dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len, dm
        )

        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        Forward pass for the decoder

        Args:
            x (tf.Tensor): (batch, target_seq_len)
            encoder_output (tf.Tensor): (batch, input_seq_len, dm)
            training (bool): training mode flag
            look_ahead_mask: mask for masked self-attention
            padding_mask: mask for encoder-decoder attention

        Returns:
            tf.Tensor: (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, encoder_output, training,
                look_ahead_mask, padding_mask
            )

        return x
