import numpy as np
import torch

from ox4rl.dataset.atari_data_collector import AtariDataCollector
from ox4rl.dataset.atari_labels import get_moving_indices, label_list_for
from ox4rl.utils.bbox_matching import compute_iou, compute_misalignment, compute_hits, match_bounding_boxes_center_distances

THRESHOLDS = {
    "AP_IOU": np.linspace(0.1, 0.9, 9),
    "PREC_REC": np.append(np.arange(0.5, 0.95, 0.05), np.arange(0.95, 1.0, 0.01))
}

class ApAndAccEval():

    def __init__(self, cfg):
        self.cfg = cfg
        self.game = cfg.gamelist[0]

    @torch.no_grad()
    def eval_ap_and_acc(self, data_subset_modes: list, dataset_mode: str) -> dict:
        """
        Evaluate average precision and accuracy
        :param logs: the model output
        :param dataset: the dataset for accessing label information
        :param bb_path: directory containing the gt bounding boxes.
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        print('Computing error rates, counts and APs...')
        data = self.get_bbox_data_via_dataloader(data_subset_modes, dataset_mode)
        results = self.compute_metrics(data)
        return results
    
    def get_bbox_data_via_dataloader(self, data_subset_modes, dataset_mode):
        data = {}
        for boxes_subset in data_subset_modes:
            data[boxes_subset] = AtariDataCollector.collect_bbox_data(self.cfg, dataset_mode, boxes_subset, only_collect_first_image_of_consecutive_frames=False)
        return data

    def compute_metrics(self, data):
        result = {}
        complete_label_list = label_list_for(self.game)
        labels = {"all": np.arange(len(complete_label_list)), "relevant": get_moving_indices(self.game)}
        # Comparing predicted bounding boxes with ground truth
        for gt_name, (gt_boxes, pred_boxes) in data.items():
            # compute results
            error_rate, perfect, overcount, undercount = compute_counts(pred_boxes, gt_boxes)
            accuracy = perfect / (perfect + overcount + undercount)
            # A list of length 9 and P/R from low IOU level = 0.2s
            try:
                aps = compute_ap(pred_boxes, gt_boxes, THRESHOLDS["AP_IOU"])
            except Exception as e:
                print(f"Failed to compute AP: {e}")
                aps = []

            precision, recall, precisions, recalls = compute_prec_rec(pred_boxes, gt_boxes, THRESHOLDS["PREC_REC"])
            average_distance = compute_average_center_distances(pred_boxes, gt_boxes)

            # store results
            result[f'error_rate_{gt_name}'] = error_rate
            result[f'perfect_{gt_name}'] = perfect
            result[f'overcount_{gt_name}'] = overcount
            result[f'undercount_{gt_name}'] = undercount
            result[f'accuracy_{gt_name}'] = accuracy
            result[f'APs_{gt_name}'] = aps
            result[f'precision_{gt_name}'] = precision
            result[f'recall_{gt_name}'] = recall
            result[f'precisions_{gt_name}'] = precisions
            result[f'recalls_{gt_name}'] = recalls
            result[f'average_distance_{gt_name}'] = average_distance

            # compute recall for object types
            for label in labels[gt_name]:
                # filter boxes by label
                gt_boxes_of_label = [gt_box_arr[gt_box_arr[:, 5] == label] for gt_box_arr in gt_boxes]
                precision, recall, precisions, recalls = compute_prec_rec(pred_boxes, gt_boxes_of_label, THRESHOLDS["PREC_REC"])
                result[f'recall_label_{label}_{gt_name}'] = recall
        return result

def compute_counts(boxes_pred: list, boxes_gt: list) -> tuple:
    """
    Compute error rates, perfect number, overcount number, undercount number

    :param boxes_pred: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param boxes_gt: [[y_min, y_max, x_min, x_max] * N] * B
    :return:
        error_rate: mean of ratio of differences
        perfect: integer
        overcount: integer
        undercount: integer
    """

    # Compute lengths of predictions and ground truths
    pred_lengths = np.array([len(pred) for pred in boxes_pred])
    gt_lengths = np.array([len(gt) for gt in boxes_gt])

    # Compute error rates using NumPy vectorized operations
    error_rates = np.abs(pred_lengths - gt_lengths) / gt_lengths

    # Count perfect, overcount, and undercount cases
    perfect = np.sum(pred_lengths == gt_lengths)
    overcount = np.sum(pred_lengths > gt_lengths)
    undercount = np.sum(pred_lengths < gt_lengths)

    return np.mean(error_rates), perfect, overcount, undercount

def compute_ap(pred_boxes, gt_boxes, iou_thresholds=None, recall_values=None, matching_method=compute_iou):
    """
    Compute average precision over different iou thresholds.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :param iou_thresholds: a list of iou thresholds
    :param recall_values: a list of recall values to compute AP
    :return: AP at each iou threshold
    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.1, 0.9, 9)

    AP = []
    for threshold in iou_thresholds:
        # Compute AP for this threshold
        
        hit, count_gt = compute_hits(pred_boxes, gt_boxes, threshold, matching_method)

        if len(hit) == 0:
            AP.append(0.0)
            continue

        # Sort
        hit = sorted(hit, key=lambda x: -x[-1])
        hit = [x[0] for x in hit]
        
        # Compute average precision
        hit_cum = np.cumsum(hit)
        num_cum = np.arange(len(hit)) + 1.0
        precision = hit_cum / num_cum
        recall = hit_cum / count_gt
        
        # Compute AP at selected recall values
        precs = []
        for val in recall_values:
            prec = precision[recall >= val]
            precs.append(0.0 if prec.size == 0 else prec.max())

        # Mean over recall values
        AP.append(np.mean(precs))

    print(AP)
    return AP


def compute_prec_rec(pred_boxes, gt_boxes, threshold_values, matching_method=compute_misalignment):
    """
    Compute precision and recall using misalignment.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :return: p/r
    """
    threshold = 0.5
    hit, count_gt = compute_hits(pred_boxes, gt_boxes, threshold, matching_method)

    hit = sorted(hit, key=lambda x: -x[-1]) # sort by confidence (descending)
    confidence_values = np.array([x[1] for x in hit])
    hit = [x[0] for x in hit] # remove confidence
    hit_cum = np.cumsum(hit)
    num_cum = np.arange(len(hit)) + 1.0
    precision = hit_cum / num_cum
    recall = hit_cum / count_gt
    if len(precision) == 0 or len(recall) == 0:
        return 0.0, 0.0, np.zeros(len(threshold_values)), np.zeros(len(threshold_values))
    
    precisions_at_thresholds = np.zeros(len(threshold_values))
    recalls_at_thresholds = np.zeros(len(threshold_values))
    # store precision and recall for different confidence thresholds
    for i, thresh in enumerate(threshold_values):
        # get first index where confidence is above threshold
        index = np.argmin(confidence_values[confidence_values > thresh]) if confidence_values[confidence_values > thresh].size > 0 else None
        if index is None:
            precisions_at_thresholds[i] = np.nan
            recalls_at_thresholds[i] = np.nan
        else:
            precisions_at_thresholds[i] = precision[index]
            recalls_at_thresholds[i] = recall[index]
    return precision[-1], recall[-1], precisions_at_thresholds, recalls_at_thresholds


def compute_average_center_distances(pred_boxes, gt_boxes):
    """
    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :return: average center distance
    """

    distances = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        curr_distances = match_bounding_boxes_center_distances(gt, pred)
        distances.extend(curr_distances)
    return np.mean(distances)


