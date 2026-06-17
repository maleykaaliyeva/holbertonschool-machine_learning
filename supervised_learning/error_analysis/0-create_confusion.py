#!/usr/bin/env python3
"""Creating Confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Return confusion matrix."""
    m = labels.shape[0]
    classes = labels.shape[1]
    mat = np.zeros((classes, classes))
    labels = np.argmax(labels, axis=1)
    logits = np.argmax(logits, axis=1)

    for a, b in zip(labels, logits):
        mat[a][b] += 1
    return mat
