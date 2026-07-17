#!/usr/bin/env python3
"""Convolution with Padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    new_w = padded_w - kw + 1
    new_h = padded_h - kh + 1
    convolved = np.zeros((m, new_h, new_w))
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph),
                                      (pw, pw)),
                           mode='constant', constant_values=0)
    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] *
                                        kernel, axis=(1, 2))
    return convolved
