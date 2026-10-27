"""Classical copy-move detection: match an image's SIFT keypoints against themselves.

A copied region carries the same texture as its source, so its keypoints have near-identical
descriptors at a consistent offset. The pipeline is one small function per step:

    prepare -> extract -> match -> group -> draw_masks

`extract` is the slow step and the only one the tuning cache stores; bump EXTRACT_VERSION
whenever `prepare` or `extract` changes, so stale cached features are never reused.
"""
from dataclasses import dataclass

import cv2
import numpy as np

EXTRACT_VERSION = 1
N_NEIGHBOURS = 10
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)


@dataclass(frozen=True, eq=False)
class Features:
    """`extract`'s output for one image, plus what's needed to map back to the original."""

    points: np.ndarray      # (N, 2) float32 keypoint x, y in prepared-image px
    neighbours: np.ndarray  # (N, 10) int32 nearest other keypoints by descriptor, padded with -1
    distances: np.ndarray   # (N, 10) float32 descriptor distances, ascending, padded with inf
    scale: float            # prepared size / original size
    shape: tuple[int, int]  # original (H, W)


def prepare(image: np.ndarray, max_side: int, clahe: bool) -> tuple[np.ndarray, float]:
    """Grayscale, downscaled so the longer side is at most `max_side` (never upscaled)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(gray)
    return gray, scale


def extract(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIFT keypoints and each one's 10 nearest other keypoints by descriptor distance."""
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    n = len(keypoints)
    neighbours = np.full((n, N_NEIGHBOURS), -1, np.int32)
    distances = np.full((n, N_NEIGHBOURS), np.inf, np.float32)
    if n < 2:
        return (np.empty((0, 2), np.float32), neighbours[:0], distances[:0])
    points = np.array([kp.pt for kp in keypoints], np.float32)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, descriptors, k=min(n, N_NEIGHBOURS + 1))
    for k, row in enumerate(knn):
        others = [m for m in row if m.trainIdx != k][:N_NEIGHBOURS]
        neighbours[k, :len(others)] = [m.trainIdx for m in others]
        distances[k, :len(others)] = [m.distance for m in others]
    return points, neighbours, distances


def compute_features(image: np.ndarray, max_side: int, clahe: bool) -> Features:
    gray, scale = prepare(image, max_side, clahe)
    points, neighbours, distances = extract(gray)
    return Features(points, neighbours, distances, scale, tuple(image.shape[:2]))


def match(features: Features, ratio: float, min_distance: float) -> np.ndarray:
    """Generalized ratio test (g2NN): (M, 2) unique keypoint pairs [i, j] with i < j.

    Walking a keypoint's neighbours nearest first, neighbour i is kept while it is an exact
    duplicate (distance 0) or clearly closer than neighbour i+1 (d_i < ratio * d_i+1); the walk
    stops at the first that isn't. One keypoint can match several, so a patch pasted twice is
    found twice. Padding (inf) is never kept and gives no evidence for the one before it.
    Pairs closer than `min_distance` px are dropped: they're the same spot, not a copy.
    """
    d, d_next = features.distances[:, :-1], features.distances[:, 1:]
    clearly_closer = np.isfinite(d_next) & (d < ratio * d_next)
    passes = np.isfinite(d) & ((d == 0) | clearly_closer)
    keep = np.cumprod(passes, axis=1).astype(bool)  # stop at the first failure
    rows, cols = np.nonzero(keep)
    pairs = np.stack([rows, features.neighbours[rows, cols]], axis=1)
    if not len(pairs):
        return np.empty((0, 2), np.int64)
    gap = np.linalg.norm(features.points[pairs[:, 0]] - features.points[pairs[:, 1]], axis=1)
    pairs = np.sort(pairs[gap >= min_distance], axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)
