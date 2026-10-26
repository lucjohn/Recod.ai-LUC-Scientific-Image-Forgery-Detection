import numpy as np
import pytest

from forgery.data import build_index, load_image, load_masks
from forgery.metric import rle_decode, rle_encode


def test_counts(index):
    assert (index.source == "authentic").sum() == 2377
    assert (index.source == "forged").sum() == 2751
    assert (index.source == "supplemental").sum() == 48


def test_sample_id_unique_and_positional(index):
    assert index.sample_id.tolist() == list(range(len(index)))


def test_authentic_has_no_masks(index):
    authentic = index[index.source == "authentic"]
    assert (authentic.n_instances == 0).all()
    assert (authentic.mask_path == "").all()
    assert (index[index.source != "authentic"].n_instances >= 1).all()


def test_rgba_image_loads_as_rgb(index):
    row = index[(index.source == "forged") & (index.case_id == 57)].iloc[0]
    image = load_image(row.image_path)
    assert image.shape == (256, 320, 3)
    assert image.dtype == np.uint8


def test_masks_match_image_shape(index):
    for row in index[index.source == "forged"].head(20).itertuples():
        for mask in load_masks(row.mask_path):
            assert mask.shape == (row.height, row.width)
            assert set(np.unique(mask)) <= {0, 1}
            np.testing.assert_array_equal(rle_decode(rle_encode([mask]), mask.shape).astype(np.uint8), mask)


def test_root_without_images_raises(monkeypatch, tmp_path):
    (tmp_path / "train_images" / "authentic").mkdir(parents=True)
    monkeypatch.setenv("FORGERY_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="FORGERY_DATA_ROOT"):
        build_index()


def test_missing_root_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGERY_DATA_ROOT", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="FORGERY_DATA_ROOT"):
        build_index()
