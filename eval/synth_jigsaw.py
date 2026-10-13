"""Synthetic jigsaw scenes with known ground truth.

A random "box picture" is cut into cells; a board is drawn flat as a plain mat with
four markers around it and any pieces as the box's cells, turned as asked; the
whole is warped by a known perspective, standing in for a camera that is not
straight overhead. Because the warp and the picture are known, every stage from
calibration to identification can be checked against the truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from thirdeye.compiler.jigsaw import piece_id
from thirdeye.perception.calibrate import MarkerLayout
from thirdeye.perception.piece_match import _turn

PX = 64         # pixels per cell in the flat drawing
MARGIN = 2      # cells of white around the board
MAT = (150, 160, 150)
TILT = np.array([[0.92, 0.10, 30.0], [-0.07, 0.88, 50.0], [0.00008, 0.00012, 1.0]])


def box_picture(cols: int, rows: int, seed: int = 7, cell: int = PX) -> np.ndarray:
    """A smooth, colourful random image: distinct regions, as a real puzzle picture has."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(rows * 6, cols * 6, 3), dtype=np.uint8)
    big = cv2.resize(small, (cols * cell, rows * cell), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(big, (0, 0), 2.0)


def layout(cols: int, rows: int) -> MarkerLayout:
    return MarkerLayout.around_plate(cols, rows, side_studs=1.0, gap_studs=0.4)


def flat_board(
    box: np.ndarray,
    cols: int,
    rows: int,
    placed: Sequence[tuple[str, int, int, int]] = (),
    hidden: frozenset[int] = frozenset(),
) -> np.ndarray:
    """The board drawn top-down. `placed` is (piece id, col, row, rot) tuples."""
    size_w, size_h = (cols + 2 * MARGIN) * PX, (rows + 2 * MARGIN) * PX
    img = np.full((size_h, size_w, 3), 255, np.uint8)
    img[MARGIN * PX:(MARGIN + rows) * PX, MARGIN * PX:(MARGIN + cols) * PX] = MAT
    lay = layout(cols, rows)
    dictionary = cv2.aruco.getPredefinedDictionary(lay.dictionary)
    side = int(lay.side_studs * PX)
    for mid, (sx, sy) in lay.centres.items():
        if mid in hidden:
            continue
        cx, cy = (sx + MARGIN) * PX, (sy + MARGIN) * PX
        x0, y0 = round(cx - side / 2), round(cy - side / 2)
        marker = cv2.aruco.generateImageMarker(dictionary, mid, side)
        img[y0:y0 + side, x0:x0 + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    h, w = box.shape[:2]
    for pid, col, row, rot in placed:
        r, c = int(pid[1:pid.index("c")]), int(pid[pid.index("c") + 1:])
        cell = box[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
        cell = cv2.resize(_turn(cell, rot), (PX, PX), interpolation=cv2.INTER_AREA)
        x0, y0 = (col + MARGIN) * PX, (row + MARGIN) * PX
        img[y0:y0 + PX, x0:x0 + PX] = cell
    return img


def camera_view(flat: np.ndarray, noise: float = 0.0, gain: float = 1.0, seed: int = 0) -> np.ndarray:
    """The tilted camera's frame, with optional sensor noise and a brightness change."""
    out = cv2.warpPerspective(flat, TILT, (flat.shape[1] + 80, flat.shape[0] + 120),
                              borderValue=(255, 255, 255))
    out = np.clip(out.astype(np.float32) * gain, 0, 255)
    if noise:
        out += np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def true_ids(cols: int, rows: int) -> list[str]:
    return [piece_id(c, r) for r in range(rows) for c in range(cols)]
