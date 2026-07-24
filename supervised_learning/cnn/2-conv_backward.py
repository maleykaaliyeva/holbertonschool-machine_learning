#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Return the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (sh, sw) = stride
    (m, h_new, w_new, c_new) = dZ.shape

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant", constant_values=0)
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        A_slice = A_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_window = A_slice[vert_start:vert_end,
                                       horiz_start:horiz_end, :]

                    dA_prev_pad[i, vert_start:vert_end,
                                horiz_start:horiz_end, :] += (
                                    W[:, :, :, c] * dZ[i, h, w, c])
                    dW[:, :, :, c] += A_window * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
