import os
from typing import List, Tuple

import cv2
import numpy as np
from skimage.segmentation import slic


BoundingBox = Tuple[int, int, int, int]


class CellFinder:
    def __init__(self, model_dir):

        # Constructor for the class.

        self.model_dir = model_dir

        # Tunable parameters
        self.num_superpixels = 250
        self.num_kmeans_groups = 4
        self.min_component_area = 250
        self.max_component_area_ratio = 0.20
        self.morph_kernel_size = 5

        os.makedirs(self.model_dir, exist_ok=True)

    def train_WBC(self, train_data):
        """
        This implementation uses "classical" deterministic image processing,
        so no training is required.
        """
        pass

    def find_WBC(self, image):
       
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        # Compute superpixels
        segments = slic(
            image,
            n_segments=self.num_superpixels,
            compactness=10.0,
            sigma=1.0,
            start_label=0,
            channel_axis=-1,
        )

        unique_segments = np.unique(segments)
        cnt = len(unique_segments)

        if cnt == 0:
            return []

        # Compute mean BGR color per superpixel
        group_means = np.zeros((cnt, 3), dtype="float32")
        for specific_group in range(cnt):
            mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
            group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]

        # K-means on superpixel means
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2,
        )

        _, bestLabels, centers = cv2.kmeans(
            group_means,
            self.num_kmeans_groups,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        # Find cluster center closest to blue in BGR
        target_color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        dists = np.sqrt(np.sum((centers - target_color) ** 2, axis=1))
        target_group = int(np.argmin(dists))

        # Keep only the target cluster
        new_centers = np.zeros_like(centers, dtype=np.float32)
        new_centers[target_group] = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        # Map each superpixel to its new color
        colors_per_clump = new_centers[bestLabels.flatten()].astype("uint8")

        # Recolor the full image and convert to grayscale
        cell_mask = colors_per_clump[segments]
        cell_mask_gray = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)



        # Clean up the mask
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), dtype=np.uint8)
        cell_mask_gray = cv2.morphologyEx(cell_mask_gray, cv2.MORPH_OPEN, kernel)
        cell_mask_gray = cv2.morphologyEx(cell_mask_gray, cv2.MORPH_CLOSE, kernel)

        _, binary_mask = cv2.threshold(cell_mask_gray, 127, 255, cv2.THRESH_BINARY)




        # Connected components
        retval, labels = cv2.connectedComponents(binary_mask)





        # Bounding boxes from connected regions
        boxes: List[BoundingBox] = []
        max_component_area = int(h * w * self.max_component_area_ratio)

        for i in range(1, retval):
            coords = np.where(labels == i)
            ys = coords[0]
            xs = coords[1]

            if ys.size == 0 or xs.size == 0:
                continue

            area = ys.size
            if area < self.min_component_area:
                continue
            if area > max_component_area:
                continue

            ymin = int(np.min(ys))
            ymax = int(np.max(ys))
            xmin = int(np.min(xs))
            xmax = int(np.max(xs))

            box_h = ymax - ymin + 1
            box_w = xmax - xmin + 1

            if box_h < 12 or box_w < 12:
                continue

            aspect_ratio = box_w / float(box_h)
            if aspect_ratio < 0.35 or aspect_ratio > 2.85:
                continue

            bbox_area = box_h * box_w
            fill_ratio = area / float(bbox_area)
            if fill_ratio < 0.18:
                continue

            boxes.append((ymin, xmin, ymax, xmax))

        # Reduce duplicate overlapping boxes
        boxes = self._non_max_suppression(boxes, iou_threshold=0.25)

        return boxes






    def _non_max_suppression(self, boxes, iou_threshold=0.25):
        """
        Basic non-maximum suppression on (ymin, xmin, ymax, xmax) boxes.
        """
        if not boxes:
            return []

        boxes_np = np.array(boxes, dtype=np.float32)

        y1 = boxes_np[:, 0]
        x1 = boxes_np[:, 1]
        y2 = boxes_np[:, 2]
        x2 = boxes_np[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        order = np.argsort(areas)[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])

            inter_h = np.maximum(0.0, yy2 - yy1 + 1)
            inter_w = np.maximum(0.0, xx2 - xx1 + 1)
            inter = inter_h * inter_w

            union = areas[i] + areas[order[1:]] - inter
            iou = np.zeros_like(inter)
            valid = union > 0
            iou[valid] = inter[valid] / union[valid]

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return [boxes[idx] for idx in keep]