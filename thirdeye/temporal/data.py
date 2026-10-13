"""Precomputed features and labels on disk (FR-TMP-1, NFR-PORT-1).

Features are computed once per video and cached, so training touches only small arrays
and runs on a free GPU. The layout is the one the segmentation benchmarks use:

    features/NAME.npy      float32, (T, D)
    labels/NAME.txt        one label per line, T lines

A dataset's own verbs are mapped onto the generic vocabulary (`actions.ACTIONS`) by a
dict, so the same model and metrics serve Assembly101, EgoPER or our own recordings.
Which verbs map where is a judgement made per dataset and recorded with the run
(NFR-REP-1); nothing here assumes a dataset's labels.

Extracting the features -- a pretrained video encoder (VideoMAE / I3D, SW-6) over clips
plus hand positions from `perception/hands.py` -- is the one step that needs a model
download and a GPU; `FeatureExtractor` is its interface, and `hand_features` is the
cheap perception half.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from thirdeye.temporal.actions import INDEX


class FeatureExtractor(Protocol):
    """Anything that turns a video into (T, D) per-frame features."""

    def __call__(self, video: Path) -> np.ndarray: ...


def hand_features(hand_xy: np.ndarray, size_px: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
    """Per-frame hand features from tracked hands: normalised wrist position, hand size
    and frame-to-frame speed. `hand_xy` is (T, 2) pixels with NaN where no hand is seen."""
    w, h = frame_wh
    pos = np.nan_to_num(hand_xy / np.array([w, h]), nan=0.0)
    present = (~np.isnan(hand_xy[:, 0])).astype(np.float32)
    size = np.nan_to_num(size_px / max(w, h), nan=0.0)
    speed = np.zeros(len(pos), dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * present[1:] * present[:-1]
    return np.column_stack([pos, present, size, speed]).astype(np.float32)


def save_video(root: Path, name: str, features: np.ndarray, labels: list[str]) -> None:
    if len(features) != len(labels):
        raise ValueError(f"{name}: {len(features)} feature rows, {len(labels)} labels")
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    np.save(root / "features" / f"{name}.npy", features.astype(np.float32))
    (root / "labels" / f"{name}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def load_split(
    root: Path, mapping: Mapping[str, str] | None = None
) -> list[tuple[str, np.ndarray, list[int]]]:
    """(name, features, generic class ids) for every video under `root`.

    `mapping` sends a dataset verb to a generic one; an unmapped verb that is not already
    generic is an error, never silently `idle` -- a wrong mapping would train the model
    to ignore exactly the actions it exists to find.
    """
    out = []
    for npy in sorted((root / "features").glob("*.npy")):
        raw = (root / "labels" / f"{npy.stem}.txt").read_text(encoding="utf-8").split()
        ids = []
        for verb in raw:
            generic = (mapping or {}).get(verb, verb)
            if generic not in INDEX:
                raise KeyError(f"{npy.stem}: label {verb!r} maps to no generic action {sorted(INDEX)}")
            ids.append(INDEX[generic])
        feats = np.load(npy)
        if len(feats) != len(ids):
            raise ValueError(f"{npy.stem}: {len(feats)} feature rows but {len(ids)} labels")
        out.append((npy.stem, feats, ids))
    return out
