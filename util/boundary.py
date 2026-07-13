import cv2
import numpy as np


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    if union == 0:
        return float('nan')
    return intersection / union


def per_class_boundary_iou(
    gt, pred, num_classes, dilation_ratio=0.02, ignore_index=255, skip_background=False
):
    """Per-class Boundary IoU in [0, 1]; NaN when a class has no boundary pixels."""
    valid = gt != ignore_index
    gt = gt.copy()
    pred = pred.copy()
    gt[~valid] = 0
    pred[~valid] = 0

    class_ious = {}
    for cls in range(num_classes):
        if skip_background and cls == 0:
            continue
        gt_bin = (gt == cls).astype(np.uint8)
        pred_bin = (pred == cls).astype(np.uint8)
        biou = boundary_iou(gt_bin, pred_bin, dilation_ratio=dilation_ratio)
        class_ious[cls] = float('nan') if np.isnan(biou) else float(biou)
    return class_ious
