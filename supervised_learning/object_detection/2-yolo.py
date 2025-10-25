#!/usr/bin/env python3
"""YOLO v3 object detector – adds filter_boxes"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.

    Public instance attributes:
      - model: the Darknet Keras model
      - class_names: list of class names
      - class_t: box score threshold for initial filtering
      - nms_t: IOU threshold for non-max suppression
      - anchors: anchor boxes (outputs, anchor_boxes, 2)
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor (same as 0-yolo / 1-yolo)."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [ln.strip() for ln in f if ln.strip()]

        self.class_t = float(class_t)
        self.nms_t = float(nms_t)

        if not isinstance(anchors, np.ndarray) or anchors.ndim != 3 or anchors.shape[-1] != 2:
            raise ValueError("anchors must be a numpy.ndarray of shape (outputs, anchor_boxes, 2)")
        self.anchors = anchors

    # ---------------------------
    # From 1-yolo.py (unchanged):
    # ---------------------------
    def process_outputs(self, outputs, image_size):
        """
        Converts Darknet raw outputs into:
          boxes: list of (gh, gw, anchor_boxes, 4) corner coords scaled to image
          box_confidences: list of (gh, gw, anchor_boxes, 1)
          box_class_probs: list of (gh, gw, anchor_boxes, classes)

        outputs: list of YOLO head tensors, each (gh, gw, anchor_boxes, 85)
        image_size: np.array([image_h, image_w])
        """
        image_h, image_w = image_size
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            gh, gw, na, _ = output.shape

            # t_xy, t_wh from network
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            obj = output[..., 4:5]
            cls = output[..., 5:]

            # Sigmoid for xy, objectness, and class logits (YOLOv3 uses logistic)
            sigmoid = lambda x: 1. / (1. + np.exp(-x))
            b_xy = sigmoid(t_xy)
            box_conf = sigmoid(obj)
            class_probs = sigmoid(cls)

            # Grid offsets
            cx = np.arange(gw)
            cy = np.arange(gh)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = cx_grid[..., np.newaxis]  # (gh, gw, 1)
            cy_grid = cy_grid[..., np.newaxis]  # (gh, gw, 1)

            # Normalize to [0,1] within image, then scale to pixels
            b_x = (b_xy[..., 0] + cx_grid) / gw
            b_y = (b_xy[..., 1] + cy_grid) / gh

            # Anchors for this scale
            anchor_w = self.anchors[i, :, 0]  # (na,)
            anchor_h = self.anchors[i, :, 1]  # (na,)
            anchor_w = anchor_w.reshape((1, 1, na))
            anchor_h = anchor_h.reshape((1, 1, na))

            # Width/height in normalized units
            b_w = (anchor_w * np.exp(t_wh[..., 0])) / input_w
            b_h = (anchor_h * np.exp(t_wh[..., 1])) / input_h

            # Convert to corner coordinates in pixel space
            x1 = (b_x - b_w / 2.) * image_w
            y1 = (b_y - b_h / 2.) * image_h
            x2 = (b_x + b_w / 2.) * image_w
            y2 = (b_y + b_h / 2.) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)  # (gh, gw, na, 4)

            boxes.append(box)
            box_confidences.append(box_conf)       # (gh, gw, na, 1)
            box_class_probs.append(class_probs)    # (gh, gw, na, classes)

        return boxes, box_confidences, box_class_probs

    # --------------------------------
    # NEW for 2-yolo.py: filter_boxes
    # --------------------------------
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Applies the class score threshold to filter boxes.

        Args:
            boxes: list of np.ndarrays, each (gh, gw, anchor_boxes, 4)
            box_confidences: list of np.ndarrays, each (gh, gw, anchor_boxes, 1)
            box_class_probs: list of np.ndarrays, each (gh, gw, anchor_boxes, classes)

        Returns:
            (filtered_boxes, box_classes, box_scores)
              - filtered_boxes: (N, 4)
              - box_classes: (N,)
              - box_scores: (N,)
        """
        f_boxes = []
        f_classes = []
        f_scores = []

        for b, conf, probs in zip(boxes, box_confidences, box_class_probs):
            # class scores per box: (gh, gw, na, classes)
            class_scores = conf * probs

            # best class and corresponding score per box
            box_classes = np.argmax(class_scores, axis=-1)     # (gh, gw, na)
            box_scores = np.max(class_scores, axis=-1)         # (gh, gw, na)

            # threshold mask
            mask = box_scores >= self.class_t                  # (gh, gw, na)

            # gather filtered arrays
            filtered_b = b[mask]                # (?, 4)
            filtered_c = box_classes[mask]      # (?)
            filtered_s = box_scores[mask]       # (?)

            if filtered_b.size:
                f_boxes.append(filtered_b)
                f_classes.append(filtered_c)
                f_scores.append(filtered_s)

        if not f_boxes:
            # No boxes passed the threshold
            return (np.zeros((0, 4)), np.array([], dtype=int), np.array([]))

        filtered_boxes = np.concatenate(f_boxes, axis=0)
        box_classes = np.concatenate(f_classes, axis=0)
        box_scores = np.concatenate(f_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
    