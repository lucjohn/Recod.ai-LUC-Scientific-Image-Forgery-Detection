"""Tune the SIFT detector on the tuning slice, then write the run configs for the chosen setting.

    python -m forgery.tune --config configs/01_sift_tune.yaml [--limit 20]

Every grid setting is scored on the tuning slice and logged to experiments/tuning/<base>.csv
(rewritten each run). Validation is never touched here: the chosen setting goes into
configs/<base>.yaml and its two stripped-down variants, which forgery.evaluate runs once.

The sweep reuses the detector's own functions but shares work between settings: features come
from the cache, `match` runs once per ratio, and `group` once per ratio at the smallest
min_inliers (see classical.prefix_groups). Results equal running each setting separately.
"""
import argparse
import itertools
import time
from pathlib import Path

import pandas as pd
import yaml

from forgery.cache import get_features
from forgery.classical import draw_masks, group, match, prefix_groups
from forgery.data import build_index, load_masks
from forgery.evaluate import git_commit, summarize
from forgery.metric import score_image
from forgery.split import REPO_ROOT, get_slice

GRID_KEYS = ("clahe", "ratio", "min_inliers", "pad")
FIXED_KEYS = ("max_side", "min_distance", "ransac", "ransac_threshold", "max_groups")
TUNING_DIR = REPO_ROOT / "experiments" / "tuning"
CONFIGS_DIR = REPO_ROOT / "configs"
SUMMARY_COLUMNS = ["score", "acc_authentic", "recall_forged", "mean_of1_detected", "n_images"]
TUNING_COLUMNS = [*GRID_KEYS, *SUMMARY_COLUMNS, "git_commit"]
TIE_TOLERANCE = 0.002


def grid_settings(grid: dict) -> list[dict]:
    """Every combination of the grid's values, in GRID_KEYS order."""
    unknown, missing = set(grid) - set(GRID_KEYS), set(GRID_KEYS) - set(grid)
    if unknown or missing:
        raise ValueError(f"Invalid grid: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}; "
                         f"expected exactly {list(GRID_KEYS)}")
    empty = [k for k in GRID_KEYS if not grid[k]]
    if empty:
        raise ValueError(f"Invalid grid: no values for {empty}")
    return [dict(zip(GRID_KEYS, values)) for values in itertools.product(*(grid[k] for k in GRID_KEYS))]


def check_tune_config(config: dict) -> None:
    """Fail before any work on a config whose sweep would differ from what its run configs run."""
    if config.get("split") != "tune":
        raise ValueError(f"Invalid tune config: split must be 'tune' (validation is never tuned on), "
                         f"got {config.get('split')!r}")
    predictor = config.get("predictor") or {}
    if predictor.get("type") != "sift":
        raise ValueError(f"Invalid tune config: forgery.tune only tunes the sift detector, got {predictor.get('type')!r}")
    fixed = predictor.get("params") or {}
    unknown, missing = set(fixed) - set(FIXED_KEYS), set(FIXED_KEYS) - set(fixed)
    if unknown or missing:
        raise ValueError(f"Invalid tune config: unknown params {sorted(unknown)}, missing params {sorted(missing)}; "
                         f"expected exactly {list(FIXED_KEYS)}")
    if fixed["ransac"] is not True:
        raise ValueError("Invalid tune config: the sweep always uses RANSAC grouping, so 'ransac' must be true")
    grid_settings(config.get("grid") or {})


def sweep(rows: pd.DataFrame, fixed: dict, grid: dict, cache: Path | None = None) -> pd.DataFrame:
    """One row per grid setting: its summary scores on `rows`."""
    settings = grid_settings(grid)
    per_image = {tuple(s.values()): [] for s in settings}
    start = time.perf_counter()
    for clahe in grid["clahe"]:
        for i, row in enumerate(rows.itertuples(), 1):
            f, gt = get_features(row, fixed["max_side"], clahe, cache), load_masks(row.mask_path)
            for ratio in grid["ratio"]:
                pairs = match(f, ratio, fixed["min_distance"])
                found = group(f.points, pairs, fixed["ransac_threshold"], min(grid["min_inliers"]), fixed["max_groups"])
                scored = {}  # settings that keep the same groups and pad give the same masks
                for m in grid["min_inliers"]:
                    groups = prefix_groups(found, m)
                    for pad in grid["pad"]:
                        key = (len(groups), pad)
                        if key not in scored:
                            pred = draw_masks(f.points, groups, f.scale, f.shape, pad)
                            scored[key] = {"n_pred": len(pred), "n_gt": len(gt), "score": score_image(pred, gt)}
                        per_image[(clahe, ratio, m, pad)].append(scored[key])
            if i % 100 == 0 or i == len(rows):
                print(f"clahe={clahe}: {i}/{len(rows)} images, {time.perf_counter() - start:.0f}s", flush=True)
    records = [{**s, **summarize(pd.DataFrame(per_image[tuple(s.values())]))} for s in settings]
    return pd.DataFrame(records, columns=[*GRID_KEYS, *SUMMARY_COLUMNS])


def choose(log: pd.DataFrame, tol: float = TIE_TOLERANCE) -> dict:
    """The best setting; among those within `tol` of it, the strictest (least likely a lucky pick)."""
    tied = log[log.score >= log.score.max() - tol]
    best = tied.sort_values(["min_inliers", "ratio", "pad", "clahe"], ascending=[False, True, True, True]).iloc[0]
    return {"clahe": bool(best.clahe), "ratio": float(best.ratio),
            "min_inliers": int(best.min_inliers), "pad": int(best.pad)}


def write_run_configs(base_name: str, fixed: dict, chosen: dict, configs_dir: Path = CONFIGS_DIR) -> list[Path]:
    """The chosen setting and its two variants, each differing from it in one param."""
    paths = []
    for suffix, change in (("", {}), ("_single_ransac", {"max_groups": 1}), ("_no_ransac", {"ransac": False})):
        name = base_name + suffix
        config = {"name": name, "split": "val",
                  "predictor": {"type": "sift", "params": {**fixed, **chosen, **change}}}
        path = Path(configs_dir) / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep a detector grid on the tuning slice.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="tune on a random subset of this many images (smoke test)")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    check_tune_config(config)  # fail before any work
    base = config["name"].removesuffix("_tune")
    smoke = args.limit is not None
    fixed = config["predictor"]["params"]
    commit = git_commit()  # recorded now: the code that runs, even if the repo changes mid-run

    rows = get_slice(config["split"], build_index())
    if args.limit is not None and args.limit < len(rows):
        rows = rows.sample(n=args.limit, random_state=0)
    log = sweep(rows, fixed, config["grid"])
    log["git_commit"] = commit
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    # A --limit run is a smoke test: it must never replace the real log or the run configs.
    log_path = TUNING_DIR / f"{base}{'-smoke' if smoke else ''}.csv"
    log[TUNING_COLUMNS].to_csv(log_path, index=False)

    chosen = choose(log)
    best = log.loc[(log[list(GRID_KEYS)] == pd.Series(chosen)).all(axis=1)].iloc[0]
    print(f"wrote {log_path} ({len(log)} settings on {len(rows)} images)")
    print(f"chosen: {chosen}  score {best.score:.4f} (best {log.score.max():.4f})")
    if smoke:
        print("smoke run (--limit): run configs not written")
        return
    for path in write_run_configs(base, fixed, chosen, CONFIGS_DIR):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
