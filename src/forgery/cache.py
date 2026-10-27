"""Per-image cache of SIFT features, so tuning can sweep many settings without re-running SIFT.

Only the tuning sweep reads it; a validation run always computes features from the image.
The cache lives next to the data (outside the repo and OneDrive), one folder per extraction
version and setting, so a change to `prepare`/`extract` never mixes old files with new.
"""
import os
from pathlib import Path

import numpy as np

from forgery import classical
from forgery.classical import Features, compute_features
from forgery.data import data_root, load_image


def cache_root() -> Path:
    return data_root().parent / "cache"


def cache_dir(max_side: int, clahe: bool, root: Path | None = None) -> Path:
    root = cache_root() if root is None else Path(root)
    return root / f"sift-v{classical.EXTRACT_VERSION}-side{max_side}-clahe{int(clahe)}"


def save_features(path: Path, f: Features) -> None:
    """Write atomically: an interrupted build never leaves a half-written file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as out:
        np.savez(out, points=f.points, neighbours=f.neighbours, distances=f.distances,
                 scale=f.scale, shape=np.array(f.shape))
    os.replace(tmp, path)


def load_features(path: Path) -> Features:
    with np.load(path) as z:
        return Features(z["points"], z["neighbours"], z["distances"], float(z["scale"]),
                        tuple(int(v) for v in z["shape"]))


def get_features(row, max_side: int, clahe: bool, root: Path | None = None) -> Features:
    """Features for an index row (needs source, case_id, image_path), computed on a cache miss."""
    path = cache_dir(max_side, clahe, root) / f"{row.source}_{row.case_id}.npz"
    if path.exists():
        return load_features(path)
    f = compute_features(load_image(row.image_path), max_side, clahe)
    save_features(path, f)
    return f
