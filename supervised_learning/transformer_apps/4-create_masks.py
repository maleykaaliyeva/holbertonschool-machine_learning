#!/usr/bin/env python3
"""
Creates all masks for Transformer training and validation
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
                containing the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out)
                containing the target sentence

    Returns:
        encoder_mask: padding mask for the encoder
        combined_mask: lookahead & padding mask for 1st decoder attention block
        decoder_mask: padding mask for 2nd decoder attention block
    """
    # Create padding mask for encoder inputs
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Create padding mask for decoder inputs (same as encoder mask)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Create padding mask for target sequence
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Create lookahead mask for target sequence
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Combine target padding mask and lookahead mask
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
