#!/usr/bin/env python3
"""YOLO v3 object detection module."""

import cv2
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

    def non_max_suppression(
            self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            indices = np.where(box_classes == cls)[0]

            boxes = filtered_boxes[indices]
            scores = box_scores[indices]

            order = np.argsort(scores)[::-1]

            while len(order) > 0:
                best = order[0]

                box_predictions.append(boxes[best])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(scores[best])

                if len(order) == 1:
                    break

                x1 = np.maximum(
                    boxes[best, 0],
                    boxes[order[1:], 0]
                )
                y1 = np.maximum(
                    boxes[best, 1],
                    boxes[order[1:], 1]
                )
                x2 = np.minimum(
                    boxes[best, 2],
                    boxes[order[1:], 2]
                )
                y2 = np.minimum(
                    boxes[best, 3],
                    boxes[order[1:], 3]
                )

                intersection_width = np.maximum(0, x2 - x1)
                intersection_height = np.maximum(0, y2 - y1)

                intersection_area = (
                    intersection_width * intersection_height
                )

                best_area = (
                    (boxes[best, 2] - boxes[best, 0]) *
                    (boxes[best, 3] - boxes[best, 1])
                )

                other_areas = (
                    (boxes[order[1:], 2] - boxes[order[1:], 0]) *
                    (boxes[order[1:], 3] - boxes[order[1:], 1])
                )

                union_area = (
                    best_area + other_areas - intersection_area
                )

                iou = intersection_area / union_area

                keep = np.where(iou <= self.nms_t)[0]
                order = order[keep + 1]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder."""
        import os

        images = []
        image_paths = []

        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)

            if os.path.isfile(path):
                image = cv2.imread(path)

                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for the YOLO model."""
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
