#!/usr/bin/env python3
"""
RNN Decoder module for Machine Translation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decodes for machine translation using an RNN architecture with attention
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        vocab: integer representing the size of the output vocabulary
        embedding: integer representing dimensionality of embedding vector
        units: integer representing number of hidden units in RNN cell
        batch: integer representing batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Executes the decoder layer forward pass

        x: tensor of shape (batch, 1) containing previous word index
        s_prev: tensor of shape (batch, units) containing previous decoder state
        hidden_states: tensor of shape (batch, input_seq_len, units) containing
                       outputs of the encoder

        Returns:
            y, s
            y: tensor of shape (batch, vocab) containing output word predictions
            s: tensor of shape (batch, units) containing new decoder hidden state
        """
        context_vector, _ = self.attention(s_prev, hidden_states)
        x_embedded = self.embedding(x)
        
        context_vector = tf.expand_dims(context_vector, 1)
        x_concat = tf.concat([context_vector, x_embedded], axis=-1)

        output, state = self.gru(x_concat, initial_state=s_prev)
        output = tf.reshape(output, (-1, output.shape[2]))

        y = self.F(output)
        return y, state
