#!/usr/bin/env python3
"""Valid Convolution."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Return a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_w = w - kw + 1
    new_h = h - kh + 1
    convolved = np.zeros((m, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(images[:, i:i + kh, j:j + kw] *
                                        kernel, axis=(1, 2))
    return convolved
