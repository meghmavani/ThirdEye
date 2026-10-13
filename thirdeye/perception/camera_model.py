"""A pinhole camera in the Jenga table frame, and its pose from table markers (FR-CAL-8).

World frame, millimetres: the tower's base corner is the origin, x and y run along
the table (the tower occupies [0, 75] x [0, 75]), z is up. Camera frame is the
OpenCV one: x right, y down, z forward, and a depth map holds the forward distance
z. Everything in `tower_view.py` is geometry in these two frames.

Two ways to get a `PinholeCamera`:

*`look_at`* -- from a position and a target, for tests and for planning a rig.

*`pose_from_markers`* -- from a colour frame: ArUco markers laid flat on the table
around the tower base have known positions, so `solvePnP` recovers the camera's
pose. The known block size (75 x 25 x 15 mm) is what makes the result metric
without a separate scale marker (FR-CAL-8).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics `k` (3x3, pixels), rotation `r` (world -> camera) and centre `c` (mm)."""

    k: np.ndarray
    r: np.ndarray
    c: np.ndarray
    width: int
    height: int

    @classmethod
    def look_at(
        cls,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        k: np.ndarray,
        size: tuple[int, int],
    ) -> PinholeCamera:
        c = np.asarray(position, dtype=np.float64)
        f = np.asarray(target, dtype=np.float64) - c
        f /= np.linalg.norm(f)
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        return cls(np.asarray(k, dtype=np.float64), np.stack([right, down, f]), c, size[0], size[1])

    @classmethod
    def from_rvec_tvec(
        cls, k: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, size: tuple[int, int]
    ) -> PinholeCamera:
        r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        return cls(np.asarray(k, dtype=np.float64), r, -r.T @ t, size[0], size[1])

    def to_json(self) -> dict:
        return {"k": self.k.tolist(), "r": self.r.tolist(), "c": self.c.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_json(cls, data: dict) -> PinholeCamera:
        return cls(np.asarray(data["k"], dtype=np.float64), np.asarray(data["r"], dtype=np.float64),
                   np.asarray(data["c"], dtype=np.float64), int(data["width"]), int(data["height"]))

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> camera frame (n, 3)."""
        return (np.asarray(points, dtype=np.float64) - self.c) @ self.r.T

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixels (n, 2), forward depth (n,)). Points behind the
        camera have non-positive depth and meaningless pixels."""
        cam = self.to_camera(points)
        z = cam[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.k[0, 0] * cam[:, 0] / safe + self.k[0, 2]
        v = self.k[1, 1] * cam[:, 1] / safe + self.k[1, 2]
        return np.stack([u, v], axis=1), z

    def rays(self) -> np.ndarray:
        """World-frame ray directions for every pixel, scaled so a ray's length
        parameter equals forward depth. Shape (height, width, 3)."""
        us, vs = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (us - self.k[0, 2]) / self.k[0, 0]
        y = (vs - self.k[1, 2]) / self.k[1, 1]
        cam = np.stack([x, y, np.ones_like(x)], axis=-1)
        return cam @ self.r  # camera -> world is r.T; row-vector form is cam @ r

    def view_cosine(self, point: np.ndarray, normal: np.ndarray) -> float:
        """Cosine of the angle between a surface normal and the line to the camera.
        Positive means the surface faces the camera."""
        to_cam = self.c - np.asarray(point, dtype=np.float64)
        return float(np.dot(normal, to_cam) / np.linalg.norm(to_cam))


# --- pose from table markers ----------------------------------------------------------

@dataclass(frozen=True)
class TableMarkers:
    """ArUco markers lying flat on the table, upright toward world +y.

    `centres` maps marker id -> (x, y) of its centre in mm; z is 0.
    """

    centres: Mapping[int, tuple[float, float]]
    side_mm: float = 40.0
    dictionary: int = cv2.aruco.DICT_4X4_50

    @classmethod
    def around_tower(cls, offset_mm: float = 60.0, side_mm: float = 40.0) -> TableMarkers:
        """Four markers on the diagonals, clear of the 75 mm tower footprint."""
        lo, hi = -offset_mm, 75.0 + offset_mm
        return cls({0: (lo, lo), 1: (hi, lo), 2: (hi, hi), 3: (lo, hi)}, side_mm)

    def corners(self, marker_id: int) -> np.ndarray:
        """World corners in ArUco order (top-left, top-right, bottom-right, bottom-left
        as seen from above with the marker upright toward +y)."""
        cx, cy = self.centres[marker_id]
        h = self.side_mm / 2
        return np.array(
            [[cx - h, cy + h, 0.0], [cx + h, cy + h, 0.0], [cx + h, cy - h, 0.0],
             [cx - h, cy - h, 0.0]]
        )


class PoseError(Exception):
    """The camera pose could not be recovered; the caller must not read slots."""


def pose_from_markers(
    color_bgr: np.ndarray, k: np.ndarray, markers: TableMarkers, min_markers: int = 2
) -> PinholeCamera:
    """Recover the camera pose from the table markers in one colour frame.

    Raises `PoseError` rather than returning a doubtful pose: reading slots through
    a wrong pose would sample the wrong pixels, and every slot would be misjudged
    with full confidence (FR-CAL-7's principle, for the Jenga rig).
    """
    gray = color_bgr if color_bgr.ndim == 2 else cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(markers.dictionary)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    world, image = [], []
    seen = 0
    if ids is not None:
        for mid, c in zip(ids.ravel(), corners, strict=True):
            if int(mid) in markers.centres:
                world.append(markers.corners(int(mid)))
                image.append(c.reshape(4, 2))
                seen += 1
    if seen < min_markers:
        raise PoseError(f"saw {seen} of the table markers, need {min_markers}")
    obj = np.concatenate(world).astype(np.float64)
    img = np.concatenate(image).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(k, dtype=np.float64), None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise PoseError("solvePnP failed")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(k, dtype=np.float64), None)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    if rms > 3.0:
        raise PoseError(f"pose fits the markers poorly (rms {rms:.1f} px)")
    h, w = color_bgr.shape[:2]
    return PinholeCamera.from_rvec_tvec(k, rvec, tvec, (w, h))
