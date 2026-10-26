"""Grouped, stratified train/validation split, persisted to a text file.

A group is one training case_id: an authentic/forged twin pair, or a forged image
with no twin. Keeping twins on the same side prevents leakage (the forged twin is
99% the authentic image). The split file is the source of truth and is never
regenerated silently.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forgery.data import EXCLUDED_SUPPLEMENTAL_IDS, build_index

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO_ROOT / "splits" / "val_case_ids.txt"
SLICES = ("train", "val", "supplemental")


def make_split(index: pd.DataFrame, out_path: Path = SPLIT_PATH, seed: int = 42, val_frac: float = 0.2) -> set[int]:
    forged = set(index[index.source == "forged"].case_id)
    authentic = set(index[index.source == "authentic"].case_id)
    rng = np.random.default_rng(seed)
    val: set[int] = set()
    for stratum in (sorted(forged & authentic), sorted(forged - authentic)):
        n_val = round(val_frac * len(stratum))
        val.update(int(c) for c in rng.permutation(stratum)[:n_val])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{c}\n" for c in sorted(val)))
    return val


def load_split(path: Path = SPLIT_PATH) -> set[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file {path} not found. Create it with: python -m forgery.split")
    return {int(line) for line in path.read_text().split()}


def get_slice(name: str, index: pd.DataFrame, split_path: Path = SPLIT_PATH) -> pd.DataFrame:
    """Index rows for `name`: "train" or "val" (both twins together), or "supplemental"."""
    if name not in SLICES:
        raise ValueError(f"Unknown slice {name!r}; expected one of {SLICES}")
    if name == "supplemental":
        rows = index[(index.source == "supplemental") & ~index.case_id.isin(EXCLUDED_SUPPLEMENTAL_IDS)]
    else:
        in_val = index.case_id.isin(load_split(split_path))
        rows = index[(index.source != "supplemental") & (in_val if name == "val" else ~in_val)]
    return rows.copy()


if __name__ == "__main__":
    index = build_index()
    val = make_split(index)
    rows = get_slice("val", index)
    twins = val & set(index[index.source == "authentic"].case_id)
    print(f"wrote {SPLIT_PATH}")
    print(f"val cases: {len(val)} ({len(twins)} twin pairs, {len(val) - len(twins)} forged-only)")
    print(f"val images: {len(rows)} ({(rows.source == 'authentic').sum()} authentic, {(rows.source == 'forged').sum()} forged)")
