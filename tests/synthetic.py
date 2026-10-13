"""Synthetic top-down scenes with known ground truth.

A scene is drawn flat -- a stud plane at a fixed pixel scale, with the four
calibration markers where the layout says and any bricks as coloured
rectangles -- then warped by a known perspective, standing in for a camera that
is not straight overhead (FR-CAL-2). Because the warp is known, so is the true
pixel <-> stud map, and every stage can be checked against it.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from thirdeye.compiler.schema import PartRef, Pose
from thirdeye.perception.calibrate import MarkerLayout
from thirdeye.state.lego.grid import extent

PX = 30            # pixels per stud in the flat drawing
MARGIN = 4         # studs of white around the plate
STUDS = 16
LAYOUT = MarkerLayout.around_plate(STUDS, STUDS)
#: A camera tilted and turned: not perpendicular to the plate (FR-CAL-2).
TILT = np.array([[0.9, 0.12, 40.0], [-0.08, 0.85, 90.0], [0.0001, 0.00015, 1.0]])

#: BGR colours to draw bricks in -- saturated, as real bricks are.
BGR = {
    "red": (30, 30, 200), "blue": (200, 90, 20), "yellow": (20, 210, 230),
    "green": (40, 160, 40), "white": (235, 235, 235), "black": (25, 25, 25),
}


def _flat(hidden: frozenset[int], bricks: Sequence[tuple[PartRef, Pose]]) -> np.ndarray:
    size = (STUDS + 2 * MARGIN) * PX
    img = np.full((size, size, 3), 255, np.uint8)
    lo, hi = MARGIN * PX, (MARGIN + STUDS) * PX
    img[lo:hi, lo:hi] = (170, 170, 170)   # the plate
    dictionary = cv2.aruco.getPredefinedDictionary(LAYOUT.dictionary)
    side = int(LAYOUT.side_studs * PX)
    for mid, (sx, sy) in LAYOUT.centres.items():
        if mid in hidden:
            continue
        cx, cy = (sx + MARGIN) * PX, (sy + MARGIN) * PX
        x0, y0 = round(cx - side / 2), round(cy - side / 2)
        marker = cv2.aruco.generateImageMarker(dictionary, mid, side)
        img[y0:y0 + side, x0:x0 + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    for part, pose in bricks:
        w, h = extent(part, pose.rot)
        x0, y0 = (pose.x + MARGIN) * PX, (pose.y + MARGIN) * PX
        # One pixel in from the cell edge, as real neighbouring bricks leave a seam.
        img[y0 + 1:y0 + h * PX - 1, x0 + 1:x0 + w * PX - 1] = BGR[part.color]
    return img


def scene(
    hidden: frozenset[int] = frozenset(),
    shift: tuple[float, float] = (0, 0),
    bricks: Sequence[tuple[PartRef, Pose]] = (),
) -> np.ndarray:
    """A BGR frame as the tilted camera sees it."""
    m = TILT.copy()
    m[0, 2] += shift[0]
    m[1, 2] += shift[1]
    flat = _flat(hidden, bricks)
    return cv2.warpPerspective(flat, m, (flat.shape[1] + 120, flat.shape[0] + 160),
                               borderValue=(255, 255, 255))


def to_pixels(stud: tuple[float, float]) -> tuple[float, float]:
    """Where the tilted camera sees a stud point -- the ground truth."""
    flat = np.array([[[(stud[0] + MARGIN) * PX, (stud[1] + MARGIN) * PX]]], dtype=np.float64)
    px = cv2.perspectiveTransform(flat, TILT)[0, 0]
    return float(px[0]), float(px[1])


def brick_box(part: PartRef, pose: Pose) -> tuple[float, float, float, float]:
    """The pixel box a perfect detector would draw around this brick."""
    w, h = extent(part, pose.rot)
    pts = np.array([to_pixels(c) for c in [(pose.x, pose.y), (pose.x + w, pose.y),
                                           (pose.x + w, pose.y + h), (pose.x, pose.y + h)]])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))
