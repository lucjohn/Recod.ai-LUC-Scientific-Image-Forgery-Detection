"""Score a predictor on a data slice and log the result.

    python -m forgery.evaluate --config configs/00_all_authentic.yaml [--split supplemental] [--limit 50]

Each run writes experiments/runs/<timestamp>_<name>/ (per-image scores + the config used)
and appends one summary row to experiments/results.csv, the project's lab notebook.
"""
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forgery.data import build_index, load_image, load_masks
from forgery.metric import score_image
from forgery.predictors import Predictor, Sample, build_predictor
from forgery.split import REPO_ROOT, get_slice

RESULTS_PATH = REPO_ROOT / "experiments" / "results.csv"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
RESULT_COLUMNS = ["timestamp", "name", "git_commit", "split", "n_images", "score",
                  "acc_authentic", "recall_forged", "mean_of1_detected", "runtime_s"]
# Untracked files under these folders are code a run may depend on, so they make it "dirty".
CODE_DIRS = ("src/", "configs/")


def validate_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray]:
    """Check a predictor's output; bool masks are converted to uint8."""
    checked = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask)
        if mask.shape != tuple(shape):
            raise ValueError(f"Predicted mask {i} has shape {mask.shape}, expected {tuple(shape)}")
        if mask.dtype != bool and not np.isin(mask, (0, 1)).all():
            raise ValueError(f"Predicted mask {i} must be 0/1, found values {np.unique(mask)[:5].tolist()}")
        checked.append(mask.astype(np.uint8))
    return checked


def score_slice(predictor: Predictor, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-image scores: sample_id, case_id, source, n_pred, n_gt, score."""
    records = []
    for row in rows.itertuples():
        image = load_image(row.image_path)
        pred = validate_masks(predictor.predict(Sample(row.sample_id, image)), image.shape[:2])
        gt = load_masks(row.mask_path)
        records.append({
            "sample_id": row.sample_id,
            "case_id": row.case_id,
            "source": row.source,
            "n_pred": len(pred),
            "n_gt": len(gt),
            "score": score_image(pred, gt),
        })
    return pd.DataFrame(records)


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def summarize(per_image: pd.DataFrame) -> dict:
    """Official score plus the breakdown that explains it (NaN when a group is empty)."""
    authentic = per_image.n_gt == 0
    flagged_forged = ~authentic & (per_image.n_pred > 0)
    return {
        "n_images": len(per_image),
        "score": _mean(per_image.score),
        "acc_authentic": _mean((per_image.n_pred[authentic] == 0).astype(float)),
        "recall_forged": _mean((per_image.n_pred[~authentic] > 0).astype(float)),
        "mean_of1_detected": _mean(per_image.score[flagged_forged]),
    }


def git_commit(cwd: Path = REPO_ROOT) -> str:
    """Short HEAD hash, with "-dirty" if tracked files changed or new code files are uncommitted.

    Untracked files elsewhere (notes, specs, run outputs) don't affect what ran, so they're ignored.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd,
                                capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = any(not line.startswith("??") or line[3:].startswith(CODE_DIRS) for line in status)
    return f"{sha}-dirty" if dirty else sha


def check_config(config: dict, split: str) -> None:
    """Fail fast, before any scoring, on a config the runner can't record."""
    if not config.get("name"):
        raise ValueError("Invalid config: 'name' is required")
    if not (config.get("predictor") or {}).get("type"):
        raise ValueError("Invalid config: 'predictor.type' is required")
    if split not in ("train", "val", "supplemental"):
        raise ValueError(f"Invalid config: unknown split {split!r}")


def check_results_header(results_path: Path) -> None:
    """Refuse to append to a results file whose columns differ, which would misalign rows."""
    if results_path.exists():
        header = results_path.read_text().splitlines()[0].split(",")
        if header != RESULT_COLUMNS:
            raise ValueError(f"{results_path} has columns {header}, expected {RESULT_COLUMNS}; "
                             "move the old file aside before logging new runs")


def _new_run_dir(runs_dir: Path, stem: str) -> Path:
    run_dir, n = runs_dir / stem, 2
    while run_dir.exists():
        run_dir, n = runs_dir / f"{stem}-{n}", n + 1
    run_dir.mkdir(parents=True)
    return run_dir


def evaluate(config: dict, split: str | None = None, limit: int | None = None,
             results_path: Path = RESULTS_PATH, runs_dir: Path = RUNS_DIR) -> dict:
    start = time.perf_counter()
    split = split or config.get("split")
    results_path = Path(results_path)
    check_config(config, split)
    check_results_header(results_path)
    commit = git_commit()  # recorded now: the code that runs, even if the repo changes mid-run
    index = build_index()
    rows = get_slice(split, index)
    if limit is not None and limit < len(rows):
        rows = rows.sample(n=limit, random_state=0)
    predictor = build_predictor(config["predictor"], index)
    per_image = score_slice(predictor, rows)

    now = datetime.now()
    run_dir = _new_run_dir(Path(runs_dir), f"{now:%Y%m%d-%H%M%S}_{config['name']}")
    per_image.to_csv(run_dir / "per_image.csv", index=False)
    run_config = {**config, "split": split, "limit": limit}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))

    result = {
        "timestamp": now.isoformat(timespec="seconds"),
        "name": config["name"],
        "git_commit": commit,
        "split": split,
        **summarize(per_image),
        "runtime_s": round(time.perf_counter() - start, 1),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result], columns=RESULT_COLUMNS).to_csv(
        results_path, mode="a", header=not results_path.exists(), index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a predictor config on a data slice.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "supplemental"])
    parser.add_argument("--limit", type=int, help="score a random subset of this many images")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    result = evaluate(config, split=args.split, limit=args.limit)
    width = max(map(len, result))
    for key, value in result.items():
        print(f"{key:<{width}}  {value:.4f}" if isinstance(value, float) else f"{key:<{width}}  {value}")


if __name__ == "__main__":
    main()
