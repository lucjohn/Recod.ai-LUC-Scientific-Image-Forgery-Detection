import numpy as np
import pandas as pd
import pytest

from forgery.evaluate import evaluate, score_slice, summarize, validate_masks
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


def test_results_appended(tmp_path):
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
