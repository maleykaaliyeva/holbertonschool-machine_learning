#!/usr/bin/env python3
"""
Convolutional Autoencoder Module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Parameters:
    - input_dims: tuple of integers containing the dimensions of model input
    - filters: list containing number of filters for each conv layer in encoder
    - latent_dims: tuple of integers containing dimensions of latent space

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    encoded = inputs
    for f in filters:
        encoded = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoded)
    encoder = keras.Model(inputs, encoded, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    decoded = latent_inputs

    # Reverse filters list for decoder
    rev_filters = filters[::-1]

    # Process all layers except the last two
    for f in rev_filters[:-1]:
        decoded = keras.layers.Conv2D(
            f, (3, 3), activation='relu', padding='same')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    # Second to last convolution with valid padding
    decoded = keras.layers.Conv2D(
        rev_filters[-1], (3, 3), activation='relu', padding='valid')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    # Last convolution with same padding, sigmoid, and original channels
    decoded_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(decoded)

    decoder = keras.Model(latent_inputs, decoded_output, name='decoder')

    # Autoencoder
    auto_output = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_output, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
