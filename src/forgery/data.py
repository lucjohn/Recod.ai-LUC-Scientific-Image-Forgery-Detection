"""Index of every image in the competition data, plus image and mask loaders.

One row per image file. A case_id is not unique: authentic/forged twins share it,
and 4 supplemental images reuse training IDs. `sample_id` is the unique key.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Source folder layout: (source, image dir, mask dir or None), in sample_id order.
SOURCES = [
    ("authentic", "train_images/authentic", None),
    ("forged", "train_images/forged", "train_masks"),
    ("supplemental", "supplemental_images", "supplemental_masks"),
]

# Supplemental images that duplicate a training image pixel for pixel; excluded from the
# supplemental slice so they aren't scored twice.
# Overlap check (2026-10-26): supplemental IDs 20833, 24032, 31168 and 32575 also exist in
# train_images/forged, but every pair is a different image (different sizes), so none is excluded.
EXCLUDED_SUPPLEMENTAL_IDS: frozenset[int] = frozenset()


def data_root() -> Path:
    """Folder holding the unzipped competition data (env var FORGERY_DATA_ROOT overrides)."""
    return Path(os.environ.get("FORGERY_DATA_ROOT", Path.home() / "kaggle-data" / "forgery" / "raw"))


def build_index(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root is not None else data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"Data root {root} does not exist. Download the data there or set FORGERY_DATA_ROOT."
        )
    images = {}
    for source, image_dir, _ in SOURCES:
        images[source] = sorted((root / image_dir).glob("*.png"), key=lambda p: int(p.stem))
        if not images[source]:
            raise FileNotFoundError(
                f"No PNG images in {root / image_dir}. Is FORGERY_DATA_ROOT ({root}) the unzipped data folder?"
            )
    rows = []
    for source, _, mask_dir in SOURCES:
        for image_path in images[source]:
            width, height = Image.open(image_path).size
            mask_path, n_instances = "", 0
            if mask_dir is not None:
                mask_path = str(root / mask_dir / f"{image_path.stem}.npy")
                n_instances = np.load(mask_path, mmap_mode="r").shape[0]
            rows.append({
                "case_id": int(image_path.stem),
                "source": source,
                "image_path": str(image_path),
                "mask_path": mask_path,
                "height": height,
                "width": width,
                "n_instances": n_instances,
            })
    index = pd.DataFrame(rows)
    index.insert(0, "sample_id", range(len(index)))
    return index


def load_image(path: str | Path) -> np.ndarray:
    """HxWx3 uint8 RGB, whatever the PNG's mode (RGBA, grayscale, ...)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_masks(path: str) -> list[np.ndarray]:
    """One HxW uint8 0/1 array per instance; [] for an empty path (authentic image)."""
    if not path:
        return []
    return list(np.load(path).astype(np.uint8))
