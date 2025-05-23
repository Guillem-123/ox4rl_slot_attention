"""
This file contains functions to match bounding boxes and methods to compute matching scores.
"""
import numpy as np
import torch
from typing import List, Union
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# matching score calcualation methods
def distance_matrix(x, obs): # from kalman_filter.py
    distances = squareform(pdist(np.append(x, obs, axis=0)))
    return distances[len(x):, :len(x)]

def iou(bb, gt_bb): # from labels.py # ATTENTION!!!!: this is modified iou: divide by gt area not union
    """ 
    Works in the same vein like iou, but only compares to gt size not the union, as such that SPACE is not punished for
    using a larger box, but fitting alpha/encoding
    """
    inner_width = min(bb[3], gt_bb[3]) - max(bb[2], gt_bb[2])
    inner_height = min(bb[1], gt_bb[1]) - max(bb[0], gt_bb[0])
    if inner_width < 0 or inner_height < 0:
        return 0
    # bb_height, bb_width = bb[1] - bb[0], bb[3] - bb[2]
    gt_bb_height, gt_bb_width = gt_bb[1] - gt_bb[0], gt_bb[3] - gt_bb[2]
    intersection = inner_height * inner_width
    gt_area_not_union = (gt_bb_height * gt_bb_width)
    if gt_area_not_union:
        return intersection / gt_area_not_union
    else:
        print("Gt Area is zero", gt_area_not_union, intersection, bb, gt_bb)
        return 0
    
def compute_iou(pred, gt): #from ap.py
    """

    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """

    if type(pred) != torch.Tensor:
        pred = torch.Tensor(pred)
    if type(gt) != torch.Tensor:
        gt = torch.Tensor(gt)

    compute_area = lambda b: (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])

    area_pred = compute_area(pred)[:, None]
    area_gt = compute_area(gt)[None, :]
    # (M, 1, 4), (1, N, 4)
    pred = pred[:, None]
    gt = gt[None, :]
    # (M, 1) (1, N)

    # (M, N)
    top = np.maximum(pred[:, :, 0], gt[:, :, 0])
    bottom = np.minimum(pred[:, :, 1], gt[:, :, 1])
    left = np.maximum(pred[:, :, 2], gt[:, :, 2])
    right = np.minimum(pred[:, :, 3], gt[:, :, 3])

    h_inter = np.maximum(0.0, bottom - top)
    w_inter = np.maximum(0.0, right - left)
    # (M, N)
    area_inter = h_inter * w_inter
    iou = area_inter / (area_pred + area_gt - area_inter)

    return iou

def compute_misalignment(pred, gt): #from ap.py
    """
    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """
    def compute_diagonal(bbs):
        return np.sqrt((bbs[:, 1] - bbs[:, 0]) ** 2 + (bbs[:, 3] - bbs[:, 2]) ** 2)

    # Compute length of diagonal of ground truth bounding boxes
    diagonal_gt = compute_diagonal(gt)
    
    #Compute centers of bounding boxes
    center_pred = np.column_stack([(pred[:, 0] + pred[:, 1]) / 2, (pred[:, 2] + pred[:, 3]) / 2])[:, None, :]
    center_gt = np.column_stack([(gt[:, 0] + gt[:, 1]) / 2, (gt[:, 2] + gt[:, 3]) / 2])[None, :, :]
    # create array of shape (M, N, 4) with M = # predicted bbs,  N = # ground truth bbs, 4 = y,x center of pred; y,x center of gt
    delta_bbs = np.concatenate([
        np.repeat(center_pred, len(gt), axis=1),
        np.repeat(center_gt, len(pred), axis=0)
    ], axis=2)
    # essentially create bbs with center of pred and center of gt as corners
    delta_bbs[:, :, [1, 2]] = delta_bbs[:, :, [2, 1]] 
    # compute diagonal of delta_bbs or equivalently distance between center of pred and center of gt
    diagonal_pair = compute_diagonal(delta_bbs.reshape((-1, 4))).reshape((len(pred), len(gt)))  
    return np.maximum(0, 1 - diagonal_pair / diagonal_gt)

def compute_center_distance(pred, gt):
    """
    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """

    def compute_center(bbs):
        return np.column_stack([(bbs[:, 0] + bbs[:, 1]) / 2, (bbs[:, 2] + bbs[:, 3]) / 2])

    # Compute centers of bounding boxes
    center_pred = compute_center(pred)
    center_gt = compute_center(gt)
    # (M, 1, 2), (1, N, 2)
    center_pred = center_pred[:, None]
    center_gt = center_gt[None, :]
    # (M, N, 2)
    delta = center_pred - center_gt
    # (M, N)
    distance = np.sqrt(np.sum(delta ** 2, axis=2))
    return distance

def compute_center_negative_distance(pred, gt):
    """
    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """

    def compute_center(bbs):
        return np.column_stack([(bbs[:, 0] + bbs[:, 1]) / 2, (bbs[:, 2] + bbs[:, 3]) / 2])
    

    # Compute centers of bounding boxes
    center_pred = compute_center(pred)
    center_gt = compute_center(gt)
    # (M, 1, 2), (1, N, 2)
    center_pred = center_pred[:, None]
    center_gt = center_gt[None, :]
    # (M, N, 2)
    delta = center_pred - center_gt
    # (M, N)
    distance = np.sqrt(np.sum(delta ** 2, axis=2))
    return - distance

# matching methods
def match_bounding_boxes(
        labels: np.ndarray, predicted: np.ndarray, label_list: List[Union[str, int]], matching_method=compute_misalignment
        ): # from eval_model_and_classifier.py
    """
    Match bounding boxes in labels and predicted.
    :param labels: np.ndarray of shape (n, 6) where n is the number of bounding boxes
    :param predicted: np.ndarray of shape (m, 6) where m is the number of bounding boxes
    :return: actual_list, predicted_list
    """
    if type(labels) != np.ndarray:
        labels = labels.numpy()
    if type(predicted) != np.ndarray:
        predicted = predicted.numpy()

    actual_list = []
    predicted_list = []
    NOT_AN_OBJECT = float(len(label_list))
    NOT_DETECTED = float(len(label_list) + 1)
    THRESHOLD = 0.2

    # compute matching scores
    matching_scores = matching_method(predicted, labels)
    if type(matching_scores) != np.ndarray:
        matching_scores = matching_scores.numpy()

    actual_idx_used = []
    best_actual_for_each_predicted = np.argmax(matching_scores, axis=1)
    best_predicted_for_each_actual = np.argmax(matching_scores, axis=0)
    for i in range(predicted.shape[0]):
        if best_predicted_for_each_actual[best_actual_for_each_predicted[i]] == i and matching_scores[i][best_actual_for_each_predicted[i]] > THRESHOLD:
            actual_list.append(labels[best_actual_for_each_predicted[i]][5])
            predicted_list.append(predicted[i][5])
            actual_idx_used.append(best_actual_for_each_predicted[i])
        else:
            actual_list.append(NOT_AN_OBJECT)
            predicted_list.append(predicted[i][5])

    # add unmatched bounding boxes in labels
    for i in range(labels.shape[0]):
        if i not in actual_idx_used:
            actual_list.append(labels[i][5])
            predicted_list.append(NOT_DETECTED)

    return actual_list, predicted_list

def get_label_of_best_matching_gt_bbox(
        labels: np.ndarray, predicted: np.ndarray, matching_method=compute_iou
        ): # from eval_model_and_classifier.py
    """
    Match bounding boxes in labels and predicted.
    :param labels: np.ndarray of shape (n, 6) where n is the number of bounding boxes
    :param predicted: np.ndarray of shape (m, 5) where m is the number of bounding boxes
    :return: label_for_pred (np.ndarray of shape (m, 1))
    """

    # compute matching scores
    # (#pred, #gt)
    matching_scores = matching_method(predicted, labels)

    # match bounding boxes
    # (#pred, )
    gt_id_for_pred = np.argmax(matching_scores, axis=1)
    # (#pred, )
    label_for_pred = labels[gt_id_for_pred, 5]

    return label_for_pred

#def match_bounding_boxes_z_what(
#        labels: np.ndarray, predicted: np.ndarray, matching_method=compute_misalignment
#        ): # from eval_model_and_classifier.py
#    """
#    Match bounding boxes in labels and predicted.
#    :param labels: np.ndarray of shape (n, 5) where n is the number of bounding boxes
#    :param predicted: np.ndarray of shape (m, 5) where m is the number of bounding boxes
#    :return: actual_list, predicted_list
#    """
#    actual_list = []
#    predicted_list = []
#    label_idx_used = []
#    THRESHOLD = 0.2
#
#    # compute matching scores
#    matching_scores = matching_method(predicted, labels)
#
#    # match bounding boxes
#    for i in range(predicted.shape[0]):
#        max_m_score = np.max(matching_scores[i])
#        if max_m_score > THRESHOLD:
#            actual_list.append(labels[np.argmax(matching_scores[i])][4])
#            predicted_list.append(predicted[i][4:])
#            label_idx_used.append(np.argmax(matching_scores[i]))
#        else:
#            predicted_list.append(predicted[i][4:])
#            actual_list.append(0) # 0 corresponds to no_label
#
#    return actual_list, predicted_list


#def hungarian_matching(x, obs): # from kalman_filter.py: uses just euclidean distance of the center
#    cost = distance_matrix(x, obs)
#    row_ind, col_ind = linear_sum_assignment(cost)
#    return col_ind, cost[row_ind, col_ind]

#def match_bbs(gt_bbs, boxes_batch, label_list, no_match_label): # from labels.py
#    labels = []
#    for bb in boxes_batch:
#        label, max_iou = max(((gt_bb[5], iou(bb, gt_bb)) for gt_bb in gt_bbs.itertuples(index=False, name=None)),
#                             key=lambda tup: tup[1])
#        if max_iou < 0.5:
#            label = no_match_label
#        labels.append(label)
#    return torch.LongTensor([label_list.index(label) for label in labels])

def compute_hits(pred_boxes, gt_boxes, threshold, matching_method):
    count_gt = 0
    hit = []
    # For each image, determine each prediction is a hit of not
    for pred, gt in zip(pred_boxes, gt_boxes):
        count_gt += len(gt)
        # Sort predictions within an image by decreasing confidence
        pred = sorted(pred, key=lambda x: -x[-1])

        if len(gt) == 0:
            hit.extend((False, conf) for *_, conf in pred)
            continue
        if len(pred) == 0:
            continue

        M, N = len(pred), len(gt)

        # (M, 4), (M) (N, 4)

        pred = np.array(pred)
        pred, conf = pred[:, :4], pred[:, -1]
        gt = np.array(gt)

        # (M, N)
        matching_scores = matching_method(pred, gt)
        # (M,)
        best_indices = np.argmax(matching_scores, axis=1)
        # (M,)
        best_matching_scores = matching_scores[np.arange(M), best_indices]
        # (N,), thresholding results
        valid = best_matching_scores > threshold
        used = [False] * N

        for i in range(M):
            # Only count first hit
            if valid[i] and not used[best_indices[i]]:
                hit.append((True, conf[i]))
                used[best_indices[i]] = True
            else:
                hit.append((False, conf[i]))
    
    return hit, count_gt


def match_bounding_boxes_center_distances(labels: np.ndarray, predicted: np.ndarray):
    """
    Match bounding boxes in labels and predicted.
    :param labels: np.ndarray of shape (n, 6) where n is the number of bounding boxes
    :param predicted: np.ndarray of shape (m, 6) where m is the number of bounding boxes
    :return: 
    """
    if type(labels) != np.ndarray:
        labels = labels.numpy()
    if type(predicted) != np.ndarray:
        predicted = predicted.numpy()

    if len(labels) == 0 or len(predicted) == 0:
        return []

    THRESHOLD = 0.05 
    # TODO: use of threshold is problematic because very bad detections are not punished

    # compute matching scores
    matching_scores = compute_center_distance(predicted, labels)
    if type(matching_scores) != np.ndarray:
        matching_scores = matching_scores.numpy()

    distances_of_detected_objects = []

    best_actual_for_each_predicted = np.argmin(matching_scores, axis=1)
    best_predicted_for_each_actual = np.argmin(matching_scores, axis=0)
    for i in range(predicted.shape[0]):
        if best_predicted_for_each_actual[best_actual_for_each_predicted[i]] == i and matching_scores[i][best_actual_for_each_predicted[i]] < THRESHOLD:
            distances_of_detected_objects.append(matching_scores[i][best_actual_for_each_predicted[i]])
    
    return distances_of_detected_objects