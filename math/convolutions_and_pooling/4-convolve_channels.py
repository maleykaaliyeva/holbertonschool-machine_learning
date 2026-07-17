#!/usr/bin/env python3
"""Convolution with Channels."""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    if padding == 'valid':
        ph = 0
        pw = 0

    if isinstance(padding, tuple):
        ph, pw = padding

    new_w = (w + 2 * pw - kw) // sw + 1
    new_h = (h + 2 * ph - kh) // sh + 1
    convolved = np.zeros((m, new_h, new_w))
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph),
                                      (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i * sh:i * sh + kh,
                                        j * sw:j * sw + kw, :] *
                                        kernel, axis=(1, 2, 3))
    return convolved
