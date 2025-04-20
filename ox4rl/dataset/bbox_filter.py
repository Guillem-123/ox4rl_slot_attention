import numpy as np
import torch


class BBoxFilter:
    def __init__(self, iou_threshold=0.5):
        """

        """
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def filter_predictions(self, pred_boxes, gt_boxes):
        """
        """
        valid_indices = []
        filtered_boxes = []
        matched_gt_info = []

        for frame_idx, (frame_preds, frame_gts) in enumerate(zip(pred_boxes, gt_boxes)):
            # Process ground truth boxes - extract coordinates and labels
            gt_coords = []
            gt_labels = []
            for gt_box in frame_gts:
                # Parse the CSV-like format
                parts = gt_box.split(',') if isinstance(gt_box, str) else gt_box
                coords = [float(x) for x in parts[:4]]
                gt_coords.append(coords)
                gt_labels.append((parts[4], parts[5]))  # (type, label)

            # Check each predicted box against ground truths
            for box_idx, pred_box in enumerate(frame_preds):
                best_iou = 0
                best_gt = None
                best_gt_info = None

                for gt_idx, (gt_box, gt_info) in enumerate(zip(gt_coords, gt_labels)):
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_box
                        best_gt_info = gt_info

                # If IoU exceeds threshold, keep this prediction
                if best_iou >= self.iou_threshold:
                    valid_indices.append((frame_idx, box_idx))
                    filtered_boxes.append(pred_box)
                    matched_gt_info.append(best_gt_info)

        return valid_indices, filtered_boxes, matched_gt_info


def filter_z_whats_with_valid_boxes(z_whats, valid_indices):
    """
    """
    filtered_features = []

    for frame_idx, box_idx in valid_indices:
        if isinstance(z_whats[frame_idx], (list, tuple)):
            feature = z_whats[frame_idx][box_idx]
        else:
            # Handle case where z_whats is already flattened
            feature = z_whats[frame_idx]
        filtered_features.append(feature)

    return torch.stack(filtered_features) if torch.is_tensor(filtered_features[0]) else np.stack(filtered_features)