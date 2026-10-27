import pandas as pd
import pytest
import yaml

from forgery.classical import SiftDetector
from forgery.evaluate import score_slice
from forgery.split import REPO_ROOT, get_slice
from forgery.tune import GRID_KEYS, check_tune_config, choose, grid_settings, sweep, write_run_configs

TUNE_CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "01_sift_tune.yaml").read_text())
FIXED = TUNE_CONFIG["predictor"]["params"]


def _log(rows):
    """A tuning log from (clahe, ratio, min_inliers, pad, score) tuples."""
    return pd.DataFrame([dict(zip((*GRID_KEYS, "score"), r)) for r in rows])


def test_grid_settings_count_and_order():
    settings = grid_settings(TUNE_CONFIG["grid"])
    assert len(settings) == 160
    assert settings[0] == {"clahe": False, "ratio": 0.4, "min_inliers": 4, "pad": 0}


def test_grid_rejects_unknown_key():
    grid = {**TUNE_CONFIG["grid"], "min_inlier": [6]}
    with pytest.raises(ValueError, match="min_inlier"):
        grid_settings(grid)


def test_grid_rejects_missing_key():
    grid = {k: v for k, v in TUNE_CONFIG["grid"].items() if k != "pad"}
    with pytest.raises(ValueError, match="pad"):
        grid_settings(grid)


def test_choose_tie_break():
    log = _log([(False, 0.5, 6, 5, 0.600), (False, 0.5, 10, 5, 0.599), (False, 0.5, 25, 5, 0.590)])
    assert choose(log) == {"clahe": False, "ratio": 0.5, "min_inliers": 10, "pad": 5}
    log = _log([(False, 0.6, 10, 5, 0.600), (False, 0.4, 10, 5, 0.599)])
    assert choose(log)["ratio"] == 0.4


def test_choose_clear_winner():
    log = _log([(True, 0.7, 4, 20, 0.610), (False, 0.4, 25, 0, 0.600)])
    chosen = choose(log)
    assert chosen == {"clahe": True, "ratio": 0.7, "min_inliers": 4, "pad": 20}
    assert all(type(v) in (bool, float, int) for v in chosen.values())  # plain types, safe for YAML


def test_write_run_configs(tmp_path):
    chosen = {"clahe": True, "ratio": 0.5, "min_inliers": 10, "pad": 5}
    paths = write_run_configs("01_sift", FIXED, chosen, tmp_path)
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    assert set(configs) == {"01_sift", "01_sift_single_ransac", "01_sift_no_ransac"}
    base = configs["01_sift"]
    assert base["split"] == "val" and base["predictor"] == {"type": "sift", "params": {**FIXED, **chosen}}
    for name, change in (("01_sift_single_ransac", {"max_groups": 1}), ("01_sift_no_ransac", {"ransac": False})):
        assert configs[name]["name"] == name and configs[name]["split"] == "val"
        assert configs[name]["predictor"] == {"type": "sift", "params": {**FIXED, **chosen, **change}}


def test_sweep_matches_evaluate(index, tmp_path):
    tune = get_slice("tune", index)
    rows = pd.concat([tune[tune.source == "authentic"].sample(3, random_state=0),
                      tune[tune.source == "forged"].sample(3, random_state=0)])
    setting = {"clahe": True, "ratio": 0.6, "min_inliers": 6, "pad": 5}
    log = sweep(rows, FIXED, {k: [v] for k, v in setting.items()}, cache=tmp_path)
    expected = score_slice(SiftDetector(**FIXED, **setting), rows).score.mean()
    assert len(log) == 1
    assert log.score.iloc[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("change", [
    {"params": {**FIXED, "min_inlier": 6}},
    {"params": {k: v for k, v in FIXED.items() if k != "max_groups"}},
    {"params": {**FIXED, "ransac": False}},
    {"split": "val"},
    {"grid": {**TUNE_CONFIG["grid"], "pad": []}},
], ids=["typo_param", "missing_param", "ransac_off", "val_split", "empty_grid_values"])
def test_check_tune_config_rejects(change):
    config = {**TUNE_CONFIG, "predictor": {**TUNE_CONFIG["predictor"]}}
    if "params" in change:
        config["predictor"]["params"] = change.pop("params")
    config.update(change)
    with pytest.raises(ValueError):
        check_tune_config(config)


def test_check_tune_config_accepts_committed_config():
    check_tune_config(TUNE_CONFIG)


def test_limited_run_leaves_committed_outputs_alone(tmp_path, monkeypatch):
    from forgery import tune

    monkeypatch.setattr(tune, "TUNING_DIR", tmp_path / "tuning")
    monkeypatch.setattr(tune, "CONFIGS_DIR", tmp_path / "configs")
    (tmp_path / "configs").mkdir()
    rows = pd.DataFrame({"source": ["forged"] * 3})
    monkeypatch.setattr(tune, "build_index", lambda: rows)
    monkeypatch.setattr(tune, "get_slice", lambda name, index: rows)
    fake_log = pd.DataFrame([{**s, "score": 0.5, "acc_authentic": 1.0, "recall_forged": 0.0,
                              "mean_of1_detected": float("nan"), "n_images": 2}
                             for s in grid_settings(TUNE_CONFIG["grid"])])
    monkeypatch.setattr(tune, "sweep", lambda rows, fixed, grid: fake_log)

    tune.main(["--config", str(REPO_ROOT / "configs" / "01_sift_tune.yaml"), "--limit", "2"])
    assert [p.name for p in (tmp_path / "tuning").iterdir()] == ["01_sift-smoke.csv"]
    assert list((tmp_path / "configs").iterdir()) == []
