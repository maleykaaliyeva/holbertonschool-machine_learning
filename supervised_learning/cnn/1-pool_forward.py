#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Return the output of the pooling layer"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    pooled_img = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':
                pooled_img[:, i, j, :] = np.max(A_prev[:, i * sh:i * sh + kh,
                                                       j * sw:j * sw + kw, :],
                                                axis=(1, 2))
            if mode == 'avg':
                pooled_img[:, i, j, :] = np.mean(A_prev[:, i * sh:i * sh + kh,
                                                        j * sw:j * sw + kw, :],
                                                 axis=(1, 2))
    return pooled_img
