"""The Predictor interface every pipeline version implements, and the reference baselines.

A predictor sees only a `Sample` (an opaque id and the image), never the label or the
source folder. Only the oracle baselines may read ground truth; they declare it with
`needs_ground_truth = True` and receive the index.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from forgery.data import load_masks


@dataclass(frozen=True)
class Sample:
    sample_id: int
    image: np.ndarray


class Predictor(Protocol):
    def predict(self, sample: Sample) -> list[np.ndarray]:
        """HxW uint8 0/1 masks, one per forged region; [] means authentic."""
        ...


class AllAuthentic:
    """Floor: calls every image authentic."""

    def predict(self, sample: Sample) -> list[np.ndarray]:
        return []


class OracleFullImage:
    """Perfect detection, no localisation: one whole-image mask on every forged image."""

    needs_ground_truth = True

    def __init__(self, index: pd.DataFrame):
        self.index = index

    def predict(self, sample: Sample) -> list[np.ndarray]:
        row = self.index.loc[sample.sample_id]
        if row.source == "authentic":
            return []
        return [np.ones((row.height, row.width), np.uint8)]


class OracleGT:
    """Ceiling: returns the true masks, so it must score exactly 1.0."""

    needs_ground_truth = True

    def __init__(self, index: pd.DataFrame):
        self.index = index

    def predict(self, sample: Sample) -> list[np.ndarray]:
        return load_masks(self.index.loc[sample.sample_id].mask_path)


REGISTRY = {
    "all_authentic": AllAuthentic,
    "oracle_full_image": OracleFullImage,
    "oracle_gt": OracleGT,
}


def build_predictor(cfg: dict, index: pd.DataFrame) -> Predictor:
    """Instantiate `cfg["type"]` with `cfg["params"]`; oracles also get the index."""
    try:
        cls = REGISTRY[cfg["type"]]
    except KeyError:
        raise KeyError(f"Unknown predictor type {cfg['type']!r}; choose from {sorted(REGISTRY)}") from None
    params = cfg.get("params") or {}
    if getattr(cls, "needs_ground_truth", False):
        return cls(index=index, **params)
    return cls(**params)
