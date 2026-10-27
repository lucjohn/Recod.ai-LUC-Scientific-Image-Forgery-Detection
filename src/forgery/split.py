"""Grouped, stratified train/validation split, persisted to a text file.

A group is one training case_id: an authentic/forged twin pair, or a forged image
with no twin. Keeping twins on the same side prevents leakage (the forged twin is
99% the authentic image). The split file is the source of truth and is never
regenerated silently.

The tuning slice is a second, smaller draw from the training cases only, for choosing
settings without touching validation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forgery.data import EXCLUDED_SUPPLEMENTAL_IDS, build_index

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO_ROOT / "splits" / "val_case_ids.txt"
TUNE_PATH = REPO_ROOT / "splits" / "tune_case_ids.txt"
SLICES = ("train", "val", "supplemental", "tune")


def _stratified_draw(forged: set[int], authentic: set[int], rng: np.random.Generator, frac: float) -> set[int]:
    """Draw `frac` of the twin cases and `frac` of the forged-only cases."""
    drawn: set[int] = set()
    for stratum in (sorted(forged & authentic), sorted(forged - authentic)):
        n = round(frac * len(stratum))
        drawn.update(int(c) for c in rng.permutation(stratum)[:n])
    return drawn


def _write_ids(ids: set[int], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{c}\n" for c in sorted(ids)))


def make_split(index: pd.DataFrame, out_path: Path = SPLIT_PATH, seed: int = 42, val_frac: float = 0.2) -> set[int]:
    forged = set(index[index.source == "forged"].case_id)
    authentic = set(index[index.source == "authentic"].case_id)
    val = _stratified_draw(forged, authentic, np.random.default_rng(seed), val_frac)
    _write_ids(val, out_path)
    return val


def make_tune_split(index: pd.DataFrame, val_path: Path = SPLIT_PATH, out_path: Path = TUNE_PATH,
                    seed: int = 42, frac: float = 0.25) -> set[int]:
    """Draw `frac` of the training cases (validation excluded), by the same method as make_split."""
    val = load_split(val_path)
    forged = set(index[index.source == "forged"].case_id) - val
    authentic = set(index[index.source == "authentic"].case_id) - val
    tune = _stratified_draw(forged, authentic, np.random.default_rng(seed), frac)
    _write_ids(tune, out_path)
    return tune


def load_split(path: Path = SPLIT_PATH) -> set[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found. Create it with: python -m forgery.split")
    return {int(line) for line in path.read_text().split()}


def get_slice(name: str, index: pd.DataFrame, split_path: Path = SPLIT_PATH,
              tune_path: Path = TUNE_PATH) -> pd.DataFrame:
    """Index rows for `name`: "train" or "val" (both twins together), "supplemental", or
    "tune" (the training rows whose case is in the tuning file)."""
    if name not in SLICES:
        raise ValueError(f"Unknown slice {name!r}; expected one of {SLICES}")
    if name == "supplemental":
        rows = index[(index.source == "supplemental") & ~index.case_id.isin(EXCLUDED_SUPPLEMENTAL_IDS)]
    else:
        in_val = index.case_id.isin(load_split(split_path))
        rows = index[(index.source != "supplemental") & (in_val if name == "val" else ~in_val)]
        if name == "tune":
            rows = rows[rows.case_id.isin(load_split(tune_path))]
    return rows.copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the validation split, or with --tune the tuning split.")
    parser.add_argument("--tune", action="store_true", help="write only the tuning split")
    args = parser.parse_args()
    index = build_index()
    name, path = ("tune", TUNE_PATH) if args.tune else ("val", SPLIT_PATH)
    cases = make_tune_split(index) if args.tune else make_split(index)
    rows = get_slice(name, index)
    twins = cases & set(index[index.source == "authentic"].case_id)
    print(f"wrote {path}")
    print(f"{name} cases: {len(cases)} ({len(twins)} twin pairs, {len(cases) - len(twins)} forged-only)")
    print(f"{name} images: {len(rows)} ({(rows.source == 'authentic').sum()} authentic, {(rows.source == 'forged').sum()} forged)")
