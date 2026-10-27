import subprocess

import numpy as np
import pandas as pd
import pytest

from forgery.evaluate import check_config, evaluate, git_commit, score_slice, summarize, validate_masks
from forgery.predictors import AllAuthentic, OracleFullImage, OracleGT, build_predictor
from forgery.split import get_slice

RESULT_COLUMNS = ["timestamp", "name", "git_commit", "split", "n_images", "score",
                  "acc_authentic", "recall_forged", "mean_of1_detected", "runtime_s"]


@pytest.fixture(scope="module")
def mixed_rows(index):
    val = get_slice("val", index).sort_values("sample_id")
    return pd.concat([val[val.source == "authentic"].head(15), val[val.source == "forged"].head(15)])


def test_oracle_gt_perfect(index, mixed_rows):
    assert summarize(score_slice(OracleGT(index), mixed_rows))["score"] == 1.0


def test_all_authentic_equals_authentic_share(mixed_rows):
    summary = summarize(score_slice(AllAuthentic(), mixed_rows))
    assert summary["score"] == 0.5
    assert summary["acc_authentic"] == 1.0
    assert summary["recall_forged"] == 0.0


def test_oracle_full_image_detects_all(index, mixed_rows):
    summary = summarize(score_slice(OracleFullImage(index), mixed_rows))
    assert summary["recall_forged"] == 1.0
    assert summary["acc_authentic"] == 1.0


def test_bool_masks_accepted():
    assert validate_masks([np.ones((4, 5), bool)], (4, 5))[0].dtype == np.uint8


def test_bad_masks_rejected():
    with pytest.raises(ValueError, match="shape"):
        validate_masks([np.ones((4, 6), np.uint8)], (4, 5))
    with pytest.raises(ValueError, match="0/1"):
        validate_masks([np.full((4, 5), 255, np.uint8)], (4, 5))


def test_unknown_predictor_type(index):
    with pytest.raises(KeyError, match="all_authentic"):
        build_predictor({"type": "magic", "params": {}}, index)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_commit_ignores_untracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (tmp_path / "notes.md").write_text("untracked\n")
    assert not git_commit(tmp_path).endswith("-dirty")
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_commit(tmp_path).endswith("-dirty")


def test_git_commit_flags_untracked_code(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("x\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (tmp_path / "src" / "forgery").mkdir(parents=True)
    (tmp_path / "src" / "forgery" / "new_predictor.py").write_text("x = 1\n")
    assert git_commit(tmp_path).endswith("-dirty")


CONFIG = {"name": "t", "split": "val", "predictor": {"type": "all_authentic", "params": {}}}


@pytest.mark.parametrize("broken", [
    {k: v for k, v in CONFIG.items() if k != "name"},
    {**CONFIG, "predictor": {"params": {}}},
], ids=["no_name", "no_predictor_type"])
def test_bad_config_fails_before_scoring(broken, tmp_path, monkeypatch):
    def must_not_score(*args, **kwargs):
        raise AssertionError("scored before validating the config")
    monkeypatch.setattr("forgery.evaluate.score_slice", must_not_score)
    with pytest.raises(ValueError, match="config"):
        evaluate(broken, limit=5, results_path=tmp_path / "r.csv", runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_git_commit_recorded_before_scoring(index, tmp_path, monkeypatch):
    state = {"phase": "before"}
    real_score_slice = score_slice

    def scoring_then_commit(*args, **kwargs):
        result = real_score_slice(*args, **kwargs)
        state["phase"] = "after"
        return result
    monkeypatch.setattr("forgery.evaluate.score_slice", scoring_then_commit)
    monkeypatch.setattr("forgery.evaluate.git_commit", lambda *a, **k: state["phase"])
    result = evaluate(CONFIG, limit=3, results_path=tmp_path / "r.csv", runs_dir=tmp_path / "runs")
    assert result["git_commit"] == "before"


def test_results_header_mismatch_refused(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text("timestamp,name,score\n2026-01-01,old,0.5\n")
    with pytest.raises(ValueError, match="columns"):
        evaluate(CONFIG, limit=3, results_path=results, runs_dir=tmp_path / "runs")
    assert results.read_text() == "timestamp,name,score\n2026-01-01,old,0.5\n"


def test_results_appended(index, tmp_path):
    config = {"name": "t", "split": "val", "predictor": {"type": "all_authentic", "params": {}}}
    results, runs = tmp_path / "results.csv", tmp_path / "runs"
    evaluate(config, limit=5, results_path=results, runs_dir=runs)
    evaluate(config, limit=5, results_path=results, runs_dir=runs)
    table = pd.read_csv(results)
    assert len(table) == 2
    assert list(table.columns) == RESULT_COLUMNS
    run_dirs = sorted(runs.iterdir())
    assert len(run_dirs) == 2
    for run_dir in run_dirs:
        assert (run_dir / "per_image.csv").exists()
        assert (run_dir / "config.yaml").exists()


def test_check_config_accepts_tune():
    check_config({"name": "x", "predictor": {"type": "all_authentic"}}, "tune")
