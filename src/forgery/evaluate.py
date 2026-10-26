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
    """Short HEAD hash, with "-dirty" if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _new_run_dir(runs_dir: Path, stem: str) -> Path:
    run_dir, n = runs_dir / stem, 2
    while run_dir.exists():
        run_dir, n = runs_dir / f"{stem}-{n}", n + 1
    run_dir.mkdir(parents=True)
    return run_dir


def evaluate(config: dict, split: str | None = None, limit: int | None = None,
             results_path: Path = RESULTS_PATH, runs_dir: Path = RUNS_DIR) -> dict:
    start = time.perf_counter()
    split = split or config["split"]
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
        "git_commit": git_commit(),
        "split": split,
        **summarize(per_image),
        "runtime_s": round(time.perf_counter() - start, 1),
    }
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
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
