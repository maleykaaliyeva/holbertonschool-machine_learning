#!/usr/bin/env python3
"""YOLO task 2."""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detection class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the YOLO model."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process outputs from the YOLO model."""
        image_h, image_w = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            object_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            cx = np.tile(
                np.arange(grid_w).reshape(1, grid_w, 1),
                (grid_h, 1, anchor_boxes)
            )

            cy = np.tile(
                np.arange(grid_h).reshape(grid_h, 1, 1),
                (1, grid_w, anchor_boxes)
            )

            bx = (
                1 / (1 + np.exp(-tx)) + cx
            ) / grid_w

            by = (
                1 / (1 + np.exp(-ty)) + cy
            ) / grid_h

            bw = (
                anchors[:, 0] * np.exp(tw)
            ) / self.model.input.shape[1]

            bh = (
                anchors[:, 1] * np.exp(th)
            ) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(
                np.stack((x1, y1, x2, y2), axis=-1)
            )

            box_confidences.append(
                1 / (1 + np.exp(-object_confidence))
            )

            box_class_probs.append(
                1 / (1 + np.exp(-class_probs))
            )

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on class confidence threshold."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
