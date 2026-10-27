"""Classical copy-move detection: match an image's SIFT keypoints against themselves.

A copied region carries the same texture as its source, so its keypoints have near-identical
descriptors at a consistent offset. The pipeline is one small function per step:

    prepare -> extract -> match -> group -> draw_masks

`extract` is the slow step and the only one the tuning cache stores; bump EXTRACT_VERSION
whenever `prepare` or `extract` changes, so stale cached features are never reused.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from forgery.predictors import Sample

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


def group(points: np.ndarray, pairs: np.ndarray, ransac_threshold: float, min_inliers: int,
          max_groups: int) -> list[np.ndarray]:
    """Sequential RANSAC: peel off one consistent movement (rotation, scale, shift) at a time.

    Each pair is tried in both directions, since we don't know which side is the source. A
    group is accepted if at least `min_inliers` pairs fit it; those pairs are removed and the
    search repeats. Returns one (k, 2) array of [source, copy] keypoint indices per group, in
    the order found. Seeded, so the same input always gives the same groups.
    """
    cv2.setRNGSeed(0)
    groups = []
    remaining = np.asarray(pairs, np.int64).reshape(-1, 2)
    while len(groups) < max_groups and len(remaining) >= max(min_inliers, 2):
        both = np.concatenate([remaining, remaining[:, ::-1]])
        model, inliers = cv2.estimateAffinePartial2D(
            points[both[:, 0]], points[both[:, 1]], method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
        if model is None:
            break
        inliers = inliers.ravel().astype(bool)
        forward, backward = inliers[:len(remaining)], inliers[len(remaining):]
        fits = forward | backward
        if fits.sum() < min_inliers:
            break
        groups.append(np.where(forward[:, None], remaining, remaining[:, ::-1])[fits])
        remaining = remaining[~fits]
    return groups


def prefix_groups(groups: list[np.ndarray], min_inliers: int) -> list[np.ndarray]:
    """The groups a stricter `min_inliers` would have found: the leading ones big enough.

    Valid because `group` is seeded and sequential, so tuning can run RANSAC once with the
    smallest threshold and read off every larger one.
    """
    kept = []
    for g in groups:
        if len(g) < min_inliers:
            break
        kept.append(g)
    return kept


def draw_masks(points: np.ndarray, groups: list[np.ndarray], scale: float, shape: tuple[int, int],
               pad: int, split_sides: bool = True) -> list[np.ndarray]:
    """One 0/1 layer per group at the original image size: the source hull and the copy hull,
    grown by `pad` original px. With `split_sides=False`, one hull around every endpoint."""
    h, w = shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)) if pad > 0 else None
    masks = []
    for g in groups:
        mask = np.zeros((h, w), np.uint8)
        for side in ((g[:, 0], g[:, 1]) if split_sides else (g.ravel(),)):
            xy = np.round(points[side] / scale).astype(np.int32)
            xy = np.clip(xy, 0, [w - 1, h - 1])
            cv2.fillConvexPoly(mask, cv2.convexHull(xy), 1)
        if kernel is not None:
            mask = cv2.dilate(mask, kernel)
        masks.append(mask)
    return masks


def detect(features: Features, *, ratio: float, min_distance: float, ransac: bool, ransac_threshold: float,
           min_inliers: int, max_groups: int, pad: int) -> list[np.ndarray]:
    """Masks for one image's features; [] means authentic."""
    pairs = match(features, ratio, min_distance)
    if ransac:
        groups = group(features.points, pairs, ransac_threshold, min_inliers, max_groups)
    else:  # no geometric check: every surviving match is one group
        groups = [pairs] if len(pairs) >= max(min_inliers, 1) else []
    return draw_masks(features.points, groups, features.scale, features.shape, pad, split_sides=ransac)


class SiftDetector:
    """The Chapter 1 predictor. `ransac=False` and `max_groups=1` give the two stripped-down variants."""

    def __init__(self, *, clahe: bool, ratio: float, min_inliers: int, pad: int, max_side: int = 1024,
                 min_distance: float = 10, ransac: bool = True, ransac_threshold: float = 3.0, max_groups: int = 4):
        self.max_side, self.clahe = max_side, clahe
        self.detect_params = dict(ratio=ratio, min_distance=min_distance, ransac=ransac,
                                  ransac_threshold=ransac_threshold, min_inliers=min_inliers,
                                  max_groups=max_groups, pad=pad)

    def predict(self, sample: "Sample") -> list[np.ndarray]:
        return detect(compute_features(sample.image, self.max_side, self.clahe), **self.detect_params)
