#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Return the output of the convolutional laye"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    if padding == 'valid':
        ph = 0
        pw = 0

    new_w = (w_prev + 2 * pw - kw) // sw + 1
    new_h = (h_prev + 2 * ph - kh) // sh + 1
    convolved = np.zeros((m, new_h, new_w, c_new))

    image_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0)

    for k in range(c_new):
        for i in range(new_h):
            for j in range(new_w):
                convolved[:, i, j, k] = np.sum(
                    image_padded[:, i * sh:i * sh + kh,
                                 j * sw:j * sw + kw, :] * W[:, :, :, k],
                    axis=(1, 2, 3))
    Z = convolved + b
    return activation(Z)
