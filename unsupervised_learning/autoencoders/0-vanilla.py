#!/usr/bin/env python3
"""
Vanilla Autoencoder Module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder

    Parameters:
    - input_dims: integer containing the dimensions of the model input
    - hidden_layers: list containing the number of nodes for each hidden
      layer in the encoder, respectively
    - latent_dims: integer containing the dimensions of the latent space

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent_output = keras.layers.Dense(
        latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs, latent_output, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latent_inputs, decoded_output, name='decoder')

    # Autoencoder
    auto_output = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_output, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
