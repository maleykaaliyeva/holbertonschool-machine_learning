#!/usr/bin/env python3
"""Trains an RNN Keras model using tf.data.Dataset to forecast BTC prices."""
import tensorflow as tf
from tensorflow.keras import layers, models


def create_dataset(data, input_width=24, shift=1, batch_size=32):
    """Creates a tf.data.Dataset with a sliding window.

    Args:
        data (np.ndarray): Preprocessed feature array.
        input_width (int): Window size in hours (default 24).
        shift (int): Step forecast length (default 1).
        batch_size (int): Batch size for model training.

    Returns:
        tf.data.Dataset: Windowed dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(input_width + shift, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(input_width + shift))
    ds = ds.map(lambda w: (w[:-shift], w[-1:, 3]))  # 3 is Close column index
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    """Builds and compiles an RNN model using MSE loss.

    Args:
        input_shape (tuple): Shape of the input window (time_steps, features).

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    pass
