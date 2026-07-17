#!/usr/bin/env python3
"""Same Convolution."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    # for both even and odd kernel sizes
    ph_top = kh // 2
    ph_bottom = kh - 1 - ph_top
    pw_left = kw // 2
    pw_right = kw - 1 - pw_left
    convolved = np.zeros((m, h, w))
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )
    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = np.sum(padded_images[:, i:i + kh, j:j + kw] *
                                        kernel, axis=(1, 2))
    return convolved
