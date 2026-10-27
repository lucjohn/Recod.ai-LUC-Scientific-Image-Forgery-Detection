from types import SimpleNamespace

import numpy as np
from PIL import Image

from forgery import cache, classical
from forgery.cache import cache_dir, get_features, load_features, save_features
from forgery.classical import compute_features
from test_classical import textured


def _assert_same(a, b):
    for field in ("points", "neighbours", "distances"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.scale == b.scale and a.shape == b.shape


def test_roundtrip(tmp_path):
    f = compute_features(textured(300, 200, 0), max_side=1024, clahe=False)
    save_features(tmp_path / "f.npz", f)
    _assert_same(load_features(tmp_path / "f.npz"), f)


def test_roundtrip_empty_features(tmp_path):
    f = compute_features(np.full((80, 80, 3), 128, np.uint8), max_side=1024, clahe=False)
    assert len(f.points) == 0
    save_features(tmp_path / "f.npz", f)
    _assert_same(load_features(tmp_path / "f.npz"), f)


def test_dir_depends_on_version_and_clahe(tmp_path, monkeypatch):
    assert cache_dir(1024, True, tmp_path) != cache_dir(1024, False, tmp_path)
    before = cache_dir(1024, True, tmp_path)
    monkeypatch.setattr(classical, "EXTRACT_VERSION", 2)
    assert cache_dir(1024, True, tmp_path) != before


def test_get_features_computes_then_loads(tmp_path, monkeypatch):
    image_path = tmp_path / "img.png"
    Image.fromarray(textured(120, 160, 1)).save(image_path)
    row = SimpleNamespace(source="forged", case_id=7, image_path=str(image_path))
    calls = []
    real = cache.compute_features
    monkeypatch.setattr(cache, "compute_features", lambda *a, **k: calls.append(1) or real(*a, **k))

    first = get_features(row, 1024, False, tmp_path)
    assert (cache_dir(1024, False, tmp_path) / "forged_7.npz").exists()
    second = get_features(row, 1024, False, tmp_path)
    assert len(calls) == 1
    _assert_same(first, second)
