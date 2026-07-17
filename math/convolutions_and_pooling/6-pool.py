#!/usr/bin/env python3
"""Pooling."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Return a numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1
    pooled_img = np.zeros((m, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':
                pooled_img[:, i, j, :] = np.max(images[:, i * sh:i * sh + kh,
                                                       j * sw:j * sw + kw, :],
                                                axis=(1, 2))
            if mode == 'avg':
                pooled_img[:, i, j, :] = np.mean(images[:, i * sh:i * sh + kh,
                                                        j * sw:j * sw + kw, :],
                                                 axis=(1, 2))
    return pooled_img
