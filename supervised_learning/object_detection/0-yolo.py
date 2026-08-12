#!/usr/bin/env python3
"""Defines the Yolo class."""

from tensorflow import keras as K


class Yolo:
    """Yolo v3 object detection class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a Yolo instance."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [
                line.strip() for line in f if line.strip()
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
