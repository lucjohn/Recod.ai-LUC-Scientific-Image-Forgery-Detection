import pytest

from forgery.data import EXCLUDED_SUPPLEMENTAL_IDS
from forgery.split import get_slice, load_split, make_split, make_tune_split


@pytest.fixture
def split_path(index, tmp_path):
    path = tmp_path / "val_case_ids.txt"
    make_split(index, out_path=path)
    return path


@pytest.fixture
def tune_path(index, split_path, tmp_path):
    path = tmp_path / "tune.txt"
    make_tune_split(index, val_path=split_path, out_path=path)
    return path


def _groups(index):
    forged = set(index[index.source == "forged"].case_id)
    twins = forged & set(index[index.source == "authentic"].case_id)
    return forged, twins


def test_no_case_in_both(index, split_path):
    train = get_slice("train", index, split_path)
    val = get_slice("val", index, split_path)
    assert not set(train.case_id) & set(val.case_id)
    _, twins = _groups(index)
    for case_id in set(val.case_id) & twins:
        assert set(val[val.case_id == case_id].source) == {"authentic", "forged"}


def test_reload_identical(index, tmp_path):
    path = tmp_path / "s.txt"
    assert make_split(index, out_path=path) == load_split(path)


def test_deterministic(index, tmp_path):
    assert make_split(index, out_path=tmp_path / "a.txt") == make_split(index, out_path=tmp_path / "b.txt")


def test_val_fraction(index, split_path):
    forged, _ = _groups(index)
    assert 0.18 <= len(load_split(split_path)) / len(forged) <= 0.22


def test_stratified(index, split_path):
    forged, twins = _groups(index)
    val = load_split(split_path)
    assert abs(len(val & twins) / len(val) - len(twins) / len(forged)) <= 0.03


def test_missing_split_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m forgery.split"):
        load_split(tmp_path / "x.txt")


def test_supplemental_slice(index, split_path):
    supplemental = get_slice("supplemental", index, split_path)
    assert (supplemental.source == "supplemental").all()
    assert len(supplemental) == 48 - len(EXCLUDED_SUPPLEMENTAL_IDS)
    assert not set(supplemental.case_id) & EXCLUDED_SUPPLEMENTAL_IDS


def test_unknown_slice(index, split_path):
    with pytest.raises(ValueError):
        get_slice("test", index, split_path)


def _tune(index, split_path, tune_path):
    return get_slice("tune", index, split_path, tune_path)


def test_tune_inside_train(index, split_path, tune_path):
    tune = _tune(index, split_path, tune_path)
    train = get_slice("train", index, split_path)
    assert set(tune.sample_id) <= set(train.sample_id)
    assert not set(tune.case_id) & load_split(split_path)


def test_tune_keeps_twins(index, split_path, tune_path):
    tune = _tune(index, split_path, tune_path)
    _, twins = _groups(index)
    for case_id in set(tune.case_id) & twins:
        assert set(tune[tune.case_id == case_id].source) == {"authentic", "forged"}


def test_tune_fraction(index, split_path, tune_path):
    forged, _ = _groups(index)
    train_cases = forged - load_split(split_path)
    assert 0.23 <= len(load_split(tune_path)) / len(train_cases) <= 0.27


def test_tune_deterministic(index, split_path, tmp_path):
    a = make_tune_split(index, val_path=split_path, out_path=tmp_path / "a.txt")
    b = make_tune_split(index, val_path=split_path, out_path=tmp_path / "b.txt")
    assert a == b
