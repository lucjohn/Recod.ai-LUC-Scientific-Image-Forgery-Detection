import cv2
import numpy as np
import pytest

from forgery.classical import (N_NEIGHBOURS, Features, SiftDetector, draw_masks, extract, group, match,
                               prefix_groups, prepare)
from forgery.metric import score_image
from forgery.predictors import Sample, build_predictor

P = dict(clahe=False, ratio=0.5, min_inliers=6, pad=5)


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


def copy_move(img, src_yx, dst_yx, size, gt=None):
    """Paste img[src:src+size] at dst; returns the image and the true mask (source + copy, one layer)."""
    img = img.copy()
    (sy, sx), (dy, dx) = src_yx, dst_yx
    img[dy:dy + size, dx:dx + size] = img[sy:sy + size, sx:sx + size]
    gt = np.zeros(img.shape[:2], np.uint8) if gt is None else gt.copy()
    gt[sy:sy + size, sx:sx + size] = 1
    gt[dy:dy + size, dx:dx + size] = 1
    return img, gt


def predict(image, **params):
    return SiftDetector(**params).predict(Sample(0, image))


def translations(sizes, seed=0):
    """Points and pairs for clean copies: group k moves `sizes[k]` random points by its own shift."""
    rng = np.random.default_rng(seed)
    shifts = [(300, 0), (0, 300), (300, 300), (150, 450), (450, 150)]
    src, dst = [], []
    for k, n in enumerate(sizes):
        s = rng.uniform(0, 100, (n, 2)) + 10 * k
        src.append(s)
        dst.append(s + shifts[k])
    src, dst = np.concatenate(src), np.concatenate(dst)
    points = np.concatenate([src, dst]).astype(np.float32)
    pairs = np.stack([np.arange(len(src)), np.arange(len(src)) + len(src)], axis=1)
    return points, pairs


def test_detects_copy_move():
    image, gt = copy_move(textured(400, 400, 1), (40, 40), (250, 230), 70)
    pred = predict(image, **P)
    assert len(pred) == 1
    assert score_image(pred, [gt]) > 0.5


def test_detects_patch_copied_twice():
    image, _ = copy_move(textured(400, 400, 2), (40, 40), (40, 250), 70)
    image, _ = copy_move(image, (40, 40), (250, 150), 70)
    assert predict(image, **P) != []


def test_two_copies_two_layers():
    image, _ = copy_move(textured(500, 500, 3), (20, 20), (20, 300), 60)
    image, _ = copy_move(image, (300, 20), (380, 300), 60)
    assert len(predict(image, **P)) == 2


def test_textured_no_copy_is_authentic():
    assert predict(textured(400, 400, 4), **P) == []


def test_flat_image_is_authentic():
    assert predict(np.full((200, 200, 3), 128, np.uint8), **P) == []


def test_group_caps_and_rejects():
    points, pairs = translations([20, 20, 20])
    assert [len(g) for g in group(points, pairs, 3.0, 6, 4)] == [20, 20, 20]
    assert len(group(points, pairs, 3.0, 6, 1)) == 1
    assert group(points, pairs, 3.0, 25, 4) == []


def test_group_orients_pairs():
    points, pairs = translations([20])
    flipped = pairs.copy()
    flipped[:5] = flipped[:5, ::-1]  # some pairs listed copy-first
    (g,) = group(points, flipped, 3.0, 6, 4)
    shifts = points[g[:, 1]] - points[g[:, 0]]
    assert np.allclose(shifts, shifts[0], atol=1e-3)


def test_group_deterministic():
    points, pairs = translations([20, 12, 8])
    a, b = group(points, pairs, 3.0, 4, 4), group(points, pairs, 3.0, 4, 4)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("m", [4, 10, 15, 25])
def test_prefix_matches_rerun(m):
    points, pairs = translations([20, 20, 20, 12, 8])
    expected = group(points, pairs, 3.0, m, 4)
    got = prefix_groups(group(points, pairs, 3.0, 4, 4), m)
    assert len(got) == len(expected) and all(np.array_equal(x, y) for x, y in zip(got, expected))


def _square(x, y, side=20):
    return [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]


def test_draw_masks_original_size_and_pad():
    points = np.array(_square(20, 20) + _square(100, 60), np.float32)  # prepared px
    g = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
    (tight,) = draw_masks(points, [g], scale=0.5, shape=(200, 300), pad=0)
    (padded,) = draw_masks(points, [g], scale=0.5, shape=(200, 300), pad=5)
    assert tight.shape == (200, 300) and tight.dtype == np.uint8
    assert tight[60, 60] == 1 and tight[140, 220] == 1  # source and copy, original px, one layer
    assert tight[100, 150] == 0  # the gap between them is not filled
    assert padded.sum() > tight.sum()


def test_degenerate_hull_not_empty():
    points = np.array([[10, 10], [50, 50], [10, 10], [50, 50]], np.float32)
    (mask,) = draw_masks(points, [np.array([[0, 1], [2, 3]])], scale=1.0, shape=(100, 100), pad=0)
    assert mask.sum() > 0


def test_no_ransac_variant():
    image, _ = copy_move(textured(400, 400, 1), (40, 40), (250, 230), 70)
    pred = predict(image, **P, ransac=False)
    assert len(pred) == 1
    assert pred[0][75, 75] == 1 and pred[0][285, 265] == 1


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        build_predictor({"type": "sift", "params": {**P, "min_inlier": 6}}, index=None)


def test_registry_builds_sift():
    assert isinstance(build_predictor({"type": "sift", "params": P}, index=None), SiftDetector)
