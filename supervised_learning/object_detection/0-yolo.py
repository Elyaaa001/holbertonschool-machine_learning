#!/usr/bin/env python3
"""YOLO v3 object detector - constructor only"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.

    Public instance attributes initialized:
      - model: the Darknet Keras model
      - class_names: list of class names (in index order)
      - class_t: box score threshold for initial filtering
      - nms_t: IOU threshold for non-max suppression
      - anchors: anchor boxes (outputs, anchor_boxes, 2)
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Args:
            model_path (str): path to a Darknet Keras model (.h5 / .keras).
            classes_path (str): path to text file with class names (one per line).
            class_t (float): box score threshold for the initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (np.ndarray): shape (outputs, anchor_boxes, 2) with anchor sizes.
        """
        # Load Keras Darknet model
        self.model = K.models.load_model(model_path)

        # Load class names (strip empty lines)
        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        # Store thresholds and anchors
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        if not isinstance(anchors, np.ndarray) or anchors.ndim != 3 or anchors.shape[-1] != 2:
            raise ValueError("anchors must be a numpy.ndarray of shape (outputs, anchor_boxes, 2)")
        self.anchors = anchors
