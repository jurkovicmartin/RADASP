import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_metrics(pred: np.ndarray, mask: np.ndarray, thresh: float =0.5) -> dict | None:
    """
    Compute unified set of metrics for a single anomaly or segmentation map:
    - pixel AUROC
    - pixel average precision (AP)
    - IoU (thresholded)
    - F1 (thresholded)
    - pixel accuracy (thresholded)

    Args:
        pred (np.ndarray): Prediction map.
        mask (np.ndarray): Ground truth map.
        thresh (float, optional): Threshold for anomaly map. Defaults to 0.5.

    Returns:
        dict: Dictionary of metrics or None if mask is empty.
    """
    # Empty mask
    if len(np.unique(mask)) < 2:
        return None
    
    pred_mask = threshold_pred(pred, thresh)

    return {
        "pixel_auroc": pixel_auroc(pred, mask),
        "pixel_ap": pixel_ap(pred, mask),
        "iou": compute_iou(pred_mask, mask),
        "f1": compute_f1(pred_mask, mask),
        "pixel_accuracy": pixel_accuracy(pred_mask, mask),
    }


### UTILITY FUNCTIONS

def normalize_map(map: np.ndarray) -> np.ndarray:
    """
    Normalize prediction map to range [0, 1].

    Args:
        map (np.ndarray): Prediction map.

    Returns:
        np.ndarray: Normalized prediction map.
    """
    map = map.astype(np.float32)
    # Avoid division by zero
    if map.max() == map.min():
        return np.zeros_like(map)
    
    return (map - map.min()) / (map.max() - map.min())


def threshold_pred(pred: np.ndarray, thresh :float =0.5) -> np.ndarray:
    """
    Threshold anomaly/probability map to binary mask.
    
    Args:
        pred (np.ndarray): Prediction map.
        thresh (float, optional): Threshold for anomaly map. Defaults to 0.5.

    Returns:
        np.ndarray: Thresholded prediction map.
    """
    pred = normalize_map(pred)
    return (pred >= thresh).astype(np.uint8)


### RAW PREDICTION METRICS

def pixel_auroc(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute pixel AUROC.

    Args:
        pred (np.ndarray): Prediction map.
        mask (np.ndarray): Ground truth map.

    Returns:
        float: Pixel AUROC.
    """
    pred = normalize_map(pred)
    flat_pred = pred.reshape(-1)
    flat_mask = mask.reshape(-1)

    return roc_auc_score(flat_mask, flat_pred)


def pixel_ap(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute pixel average precision (AP).

    Args:
        pred (np.ndarray): Prediction map.
        mask (np.ndarray): Ground truth map.

    Returns:
        float: Pixel average precision.
    """
    pred = normalize_map(pred)
    flat_pred = pred.reshape(-1)
    flat_mask = mask.reshape(-1)

    return average_precision_score(flat_mask, flat_pred)


### THRESHOLDED METRICS (PREDICTED MASKS)

def compute_iou(pred_mask: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU).

    Args:
        pred_mask (np.ndarray): Predicted mask.
        mask (np.ndarray): Ground truth mask.

    Returns:
        float: IoU.
    """
    intersection = np.logical_and(pred_mask, mask).sum()
    union = np.logical_or(pred_mask, mask).sum()
    return intersection / (union + 1e-8)


def compute_f1(pred_mask: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute F1 score for binary masks.

    Args:
        pred_mask (np.ndarray): Predicted mask (binary).
        mask (np.ndarray): Ground truth mask (binary).

    Returns:
        float: F1 score.
    """
    tp = np.logical_and(pred_mask, mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), mask).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)


def pixel_accuracy(pred_mask: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute pixel accuracy for binary masks.

    Args:
        pred_mask (np.ndarray): Predicted mask (binary).
        mask (np.ndarray): Ground truth mask (binary).

    Returns:
        float: Pixel accuracy.
    """
    correct_pixels = np.sum(pred_mask == mask)
    total_pixels = mask.size
    return correct_pixels / total_pixels
