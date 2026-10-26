import numpy as np
import pandas as pd
import pytest

from forgery.metric import oF1_score, rle_decode, rle_encode, score, score_image

# The five doctest examples from Kaggle's official metric notebook (metric/recodai-f1), verbatim.
OFFICIAL_EXAMPLES = [
    (["authentic", "authentic", "authentic"], ["authentic", "authentic", "authentic"],
     ["authentic", "authentic", "authentic"], 1.0),
    (["authentic", "authentic", "authentic"], ["authentic", "authentic", "authentic"],
     ["[101, 102]", "[101, 102]", "[101, 102]"], 0.0),
    (["[101, 102]", "[101, 102]", "[101, 102]"], ["[720, 960]", "[720, 960]", "[720, 960]"],
     ["[101, 102]", "[101, 102]", "[101, 102]"], 1.0),
    (["[101, 103]", "[101, 102]", "[101, 102]"], ["[720, 960]", "[720, 960]", "[720, 960]"],
     ["[101, 102]", "[101, 102]", "[101, 102]"], 0.9983739837398374),
    (["[101, 102];[300, 100]", "[101, 102]", "[101, 102]"], ["[720, 960]", "[720, 960]", "[720, 960]"],
     ["[101, 102]", "[101, 102]", "[101, 102]"], 0.8333333333333334),
]


@pytest.mark.parametrize("labels,shapes,preds,expected", OFFICIAL_EXAMPLES)
def test_official_examples(labels, shapes, preds, expected):
    solution = pd.DataFrame({"row_id": [0, 1, 2], "annotation": labels, "shape": shapes})
    submission = pd.DataFrame({"row_id": [0, 1, 2], "annotation": preds})
    assert score(solution, submission, row_id_column_name="row_id") == pytest.approx(expected, abs=1e-12)


def _random_mask():
    return (np.random.default_rng(0).random((5, 7)) > 0.5).astype(np.uint8)


def _single_pixel():
    m = np.zeros((5, 7), np.uint8)
    m[4, 6] = 1
    return m


@pytest.mark.parametrize("mask", [
    np.zeros((5, 7), np.uint8),
    _single_pixel(),
    np.ones((5, 7), np.uint8),
    _random_mask(),
], ids=["empty", "single_pixel", "full", "random"])
def test_rle_roundtrip(mask):
    decoded = rle_decode(rle_encode([mask]), mask.shape)
    np.testing.assert_array_equal(decoded.astype(np.uint8), mask)


def test_rle_is_column_major():
    m = np.zeros((2, 3), np.uint8)
    m[1, 0] = 1
    assert rle_encode([m]) == "[2, 1]"


def test_excess_prediction_penalty():
    a = np.zeros((4, 4), np.uint8)
    a[:2, :2] = 1
    b = np.zeros((4, 4), np.uint8)
    b[2:, 2:] = 1
    assert oF1_score([a, b], [a]) == pytest.approx(0.5)


def test_score_image_authentic_rules():
    a = np.zeros((4, 4), np.uint8)
    a[:2, :2] = 1
    assert score_image([], []) == 1.0
    assert score_image([a], []) == 0.0
    assert score_image([], [a]) == 0.0
    assert score_image([a], [a]) == 1.0
