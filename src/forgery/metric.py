"""Competition metric: a NumPy port of Kaggle's official notebook `metric/recodai-f1`.

Function names and behaviour follow the official code; only the numba JIT helpers
are replaced with vectorised NumPy. `score_image` is our mask-level convenience.

Scoring per image:
- if the label or the prediction is "authentic": 1 when both are, else 0;
- otherwise oF1: pixel-F1 between every predicted and true instance, best one-to-one
  pairing (Hungarian), mean of the paired F1s, times n_gt / max(n_pred, n_gt).
"""
import json

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize


class ParticipantVisibleError(Exception):
    pass


def _rle_runs(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Column-major, 1-indexed [start, length, start, length, ...] runs of `fg_val`."""
    flat = (x.T.flatten() == fg_val).astype(np.int8)
    edges = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    runs = np.empty(2 * len(starts), dtype=np.int64)
    runs[0::2] = starts + 1
    runs[1::2] = ends - starts
    return runs.tolist()


def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """Each mask's RLE as a JSON list, instances joined by ';'."""
    return ";".join(json.dumps(_rle_runs(x, fg_val)) for x in masks)


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    """Decode one JSON RLE string into a bool mask of `shape` (height, width)."""
    runs = np.asarray(json.loads(mask_rle), dtype=np.int64)
    starts = runs[0::2]
    if sorted(starts) != list(starts):
        raise ParticipantVisibleError("Submitted values must be in ascending order.")
    if len(runs) % 2 != 0:
        raise ParticipantVisibleError("One or more rows has an odd number of values.")
    starts = starts - 1
    ends = starts + runs[1::2]
    if np.any(ends[:-1] > starts[1:]):
        raise ParticipantVisibleError("Pixels must not be overlapping.")
    img = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = True
    return img.reshape(shape, order="F")


def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray) -> float:
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def calculate_f1_matrix(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]) -> npt.NDArray:
    """F1 for every (predicted, true) pair; zero rows pad missing predictions."""
    f1_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            f1_matrix[i, j] = calculate_f1_score(pred, gt)

    if f1_matrix.shape[0] < len(gt_masks):
        f1_matrix = np.vstack((f1_matrix, np.zeros((len(gt_masks) - len(f1_matrix), len(gt_masks)))))
    return f1_matrix


def oF1_score(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]) -> float:
    """Optimal F1: Hungarian pairing of instances, penalised for excess predictions."""
    f1_matrix = calculate_f1_matrix(pred_masks, gt_masks)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    # linear_sum_assignment discards excess predictions, so penalise them separately.
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return float(np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty)


def evaluate_single_image(label_rles: str, prediction_rles: str, shape_str: str) -> float:
    shape = json.loads(shape_str)
    labels = [rle_decode(x, shape=shape) for x in label_rles.split(";")]
    predictions = [rle_decode(x, shape=shape) for x in prediction_rles.split(";")]
    return oF1_score(predictions, labels)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Official CSV-level score. Rows are aligned by position, as in the official code."""
    df = solution.rename(columns={"annotation": "label"})
    df["prediction"] = submission["annotation"]
    authentic = (df["label"] == "authentic") | (df["prediction"] == "authentic")
    df["image_score"] = ((df["label"] == df["prediction"]) & authentic).astype(float)
    df.loc[~authentic, "image_score"] = df.loc[~authentic].apply(
        lambda row: evaluate_single_image(row["label"], row["prediction"], row["shape"]), axis=1
    )
    return float(np.mean(df["image_score"]))


def score_image(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]) -> float:
    """Score one image from masks. An empty list means "authentic"."""
    if not pred_masks or not gt_masks:
        return 1.0 if not pred_masks and not gt_masks else 0.0
    return oF1_score(pred_masks, gt_masks)
