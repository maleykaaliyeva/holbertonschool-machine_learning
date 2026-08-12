#!/usr/bin/env python3
"""YOLO v3 object detection module."""

import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo:
    """YOLO v3 class for object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO instance."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_conf = 1 / (1 + np.exp(-output[:, :, :, 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[:, :, :, 5:]))

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, [grid_height, 1, anchor_boxes])

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(c_y, [1, grid_width, anchor_boxes])

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            b_w = np.exp(t_w) * anchor_width / input_width
            b_h = np.exp(t_h) * anchor_height / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.zeros(
                (grid_height, grid_width, anchor_boxes, 4)
            )
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on class and box scores."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(box_score, axis=-1)
            box_score_max = np.max(box_score, axis=-1)

            mask = box_score_max >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score_max[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def load_images(self, folder_path):
        """Load images from a folder."""
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for YOLO model."""
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

            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores,
                   file_name):
        """Display image with boxes, class names, and scores."""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            label = "{} {:.2f}".format(
                self.class_names[box_classes[i]],
                box_scores[i]
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            cv2.imwrite(
                os.path.join('detections', file_name),
                image
            )

        cv2.destroyAllWindows()
