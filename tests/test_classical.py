import cv2
import numpy as np
import pytest

from forgery.classical import N_NEIGHBOURS, Features, extract, match, prepare


def textured(h, w, seed):
    """Blurred noise: plenty of SIFT keypoints, and no two regions alike."""
    noise = np.random.default_rng(seed).integers(0, 256, (h, w)).astype(np.uint8)
    gray = cv2.GaussianBlur(noise, (0, 0), 2)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return np.dstack([gray] * 3)


def features(points, neighbours, distances):
    """Hand-built Features: rows padded to N_NEIGHBOURS with -1 / inf."""
    n = len(points)
    nb = np.full((n, N_NEIGHBOURS), -1, np.int32)
    ds = np.full((n, N_NEIGHBOURS), np.inf, np.float32)
    for k, (row_nb, row_ds) in enumerate(zip(neighbours, distances)):
        nb[k, :len(row_nb)] = row_nb
        ds[k, :len(row_ds)] = row_ds
    return Features(np.asarray(points, np.float32), nb, ds, 1.0, (100, 100))


def test_prepare_downscales_only():
    gray, scale = prepare(np.zeros((1000, 2048, 3), np.uint8), max_side=1024, clahe=False)
    assert gray.shape == (500, 1024) and scale == 0.5
    gray, scale = prepare(np.zeros((200, 300, 3), np.uint8), max_side=1024, clahe=False)
    assert gray.shape == (200, 300) and scale == 1.0


def test_prepare_clahe_changes_dark_image():
    dark = (textured(200, 200, 0) // 9).astype(np.uint8)  # values 0-28
    plain, _ = prepare(dark, max_side=1024, clahe=False)
    boosted, _ = prepare(dark, max_side=1024, clahe=True)
    assert boosted.shape == plain.shape
    assert not np.array_equal(boosted, plain)


def test_extract_shapes():
    gray, _ = prepare(textured(300, 300, 0), max_side=1024, clahe=False)
    points, neighbours, distances = extract(gray)
    n = len(points)
    assert n > 20
    assert points.shape == (n, 2) and neighbours.shape == (n, N_NEIGHBOURS) and distances.shape == (n, N_NEIGHBOURS)
    assert not (neighbours == np.arange(n)[:, None]).any()
    assert (np.diff(distances, axis=1) >= 0).all()


def test_extract_few_keypoints():
    points, neighbours, distances = extract(np.full((100, 100), 128, np.uint8))
    assert points.shape == (0, 2) and neighbours.shape == (0, N_NEIGHBOURS) and distances.shape == (0, N_NEIGHBOURS)

    blobs = np.full((120, 300), 255, np.uint8)
    for x in (50, 150, 250):
        cv2.circle(blobs, (x, 60), 12, 0, -1)
    points, neighbours, distances = extract(blobs)
    n = len(points)
    assert 2 <= n < N_NEIGHBOURS
    assert (neighbours[:, n - 1:] == -1).all() and np.isinf(distances[:, n - 1:]).all()
    assert (neighbours[:, :n - 1] >= 0).all()


def test_match_ratio_walk():
    far = [[0, 0], [50, 0], [0, 50], [50, 50]]
    f = features(far, [[1, 2, 3], [], [], []], [[10, 100, 110], [], [], []])
    assert match(f, ratio=0.5, min_distance=10).tolist() == [[0, 1]]
    f = features(far, [[1, 2, 3], [], [], []], [[10, 15, 300], [], [], []])
    assert match(f, ratio=0.5, min_distance=10).tolist() == []


def test_match_keeps_zero_distance_neighbours():
    far = [[0, 0], [50, 0], [0, 50], [50, 50], [90, 90]]
    f = features(far, [[1, 2, 3, 4], [], [], [], []], [[0, 0, 0, 200], [], [], [], []])
    assert match(f, ratio=0.5, min_distance=10).tolist() == [[0, 1], [0, 2], [0, 3]]


def test_match_min_distance():
    pts = [[0, 0], [5, 0], [20, 0]]
    f = features(pts, [[1, 2], [], []], [[10, 100], [], []])
    assert match(f, ratio=0.5, min_distance=10).tolist() == []
    f = features(pts, [[2, 1], [], []], [[10, 100], [], []])
    assert match(f, ratio=0.5, min_distance=10).tolist() == [[0, 2]]


def test_match_unique_pairs():
    far = [[0, 0], [50, 0], [0, 90]]
    f = features(far, [[1, 2], [0, 2], []], [[10, 100], [10, 100], []])
    pairs = match(f, ratio=0.5, min_distance=10)
    assert pairs.shape == (1, 2) and pairs.tolist() == [[0, 1]]
