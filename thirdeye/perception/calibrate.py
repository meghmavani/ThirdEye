"""Top-down calibration: ArUco markers to a pixel -> stud homography (FR-CAL-1..4, FR-CAL-7).

Four printed markers sit at known stud coordinates around the baseplate. Their
centres give four point correspondences, which fix a homography exactly; from
then on `state/lego/grid.py` maps any detection onto the stud grid, whatever
angle the camera is at (FR-CAL-2).

Centres, not corners, and on purpose. Using all sixteen corners would
over-determine the fit, but it would also make calibration depend on each
marker being glued down the right way round -- a mistake nobody notices until
every brick lands a stud off. Centres are indifferent to marker rotation. The
corners are still used, as a check: a marker's side mapped through the
homography should measure its printed size in studs, and the worst deviation is
reported as `error_studs`.

**Losing calibration is reported, never papered over** (FR-CAL-7). A hand over
one corner is routine, so a missing marker alone does not invalidate the
homography: the markers that are still visible are checked against where the
last homography put them, and if they have not moved, neither has the camera.
If they have moved, or none has been seen for too long, the status is `lost`
and `homography` is None, so a caller cannot emit coordinates from a stale fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

#: `ok`: fitted this refresh from all four markers. `held`: some markers hidden,
#: the visible ones unmoved, last fit kept. `stale`: nothing visible, last fit
#: kept within the grace period. `lost`: do not use. `uncalibrated`: never fitted.
Status = Literal["ok", "held", "stale", "lost", "uncalibrated"]

USABLE: frozenset[str] = frozenset({"ok", "held", "stale"})


@dataclass(frozen=True)
class MarkerLayout:
    """Where each marker's centre sits, in stud coordinates, and its printed size.

    The default puts markers 0-3 just outside the corners of a 16 x 16 plate,
    anticlockwise from the origin. Coordinates are continuous studs, so a
    marker off the plate is simply a negative or out-of-range coordinate.
    """

    centres: Mapping[int, tuple[float, float]]
    side_studs: float = 3.0
    dictionary: int = cv2.aruco.DICT_4X4_50

    @classmethod
    def around_plate(
        cls, studs_x: int, studs_y: int, side_studs: float = 3.0, gap_studs: float = 0.5
    ) -> MarkerLayout:
        o = side_studs / 2 + gap_studs
        return cls(
            centres={
                0: (-o, -o),
                1: (studs_x + o, -o),
                2: (studs_x + o, studs_y + o),
                3: (-o, studs_y + o),
            },
            side_studs=side_studs,
        )


@dataclass(frozen=True)
class Calibration:
    """One refresh's verdict."""

    status: Status
    #: Pixel -> stud, 3x3. None whenever `status` is not usable.
    homography: np.ndarray | None
    seen: frozenset[int] = frozenset()
    #: Worst marker-side length error through the fit, in studs. How far to
    #: trust the grid; the detector's own error comes on top of it.
    error_studs: float | None = None

    @property
    def usable(self) -> bool:
        return self.status in USABLE and self.homography is not None


def fit(pixels: Sequence[tuple[float, float]], studs: Sequence[tuple[float, float]]) -> np.ndarray:
    """The homography taking four (or more) pixel points to stud points."""
    src = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(studs, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("a homography needs at least four matched points")
    if len(src) == 4:
        h = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else:
        h, _ = cv2.findHomography(src, dst)
    if h is None or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("points are degenerate (collinear or repeated)")
    return h / h[2, 2]


def from_corners(
    corners_px: Sequence[tuple[float, float]], studs_x: int, studs_y: int
) -> Calibration:
    """The manual fallback (FR-CAL-4): four clicked plate corners, anticlockwise
    from the origin corner, when the markers cannot be found."""
    studs = [(0.0, 0.0), (float(studs_x), 0.0), (float(studs_x), float(studs_y)),
             (0.0, float(studs_y))]
    return Calibration(status="ok", homography=fit(corners_px, studs), seen=frozenset())


def _map(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), h).reshape(-1, 2)


@dataclass
class Calibrator:
    """Keeps a top-down camera's homography current (FR-CAL-3).

    Call `update(frame)` every frame. Detection runs on every `refresh_every`-th
    frame only -- that is what keeps the amortised cost under NFR-PERF-6 -- plus
    the frame after any refresh that saw the markers move, so a knocked camera
    is refitted immediately rather than ten frames later.
    """

    layout: MarkerLayout
    refresh_every: int = 10
    #: A visible marker further than this from where the last fit put it means
    #: the camera or the plate moved.
    shift_px: float = 4.0
    #: Refreshes with no marker in view before the last fit is abandoned. At 30
    #: FPS and a refresh every 10 frames, 6 is two seconds.
    grace: int = 6

    current: Calibration = field(
        default_factory=lambda: Calibration(status="uncalibrated", homography=None), init=False
    )
    _frame: int = field(default=0, init=False)
    _urgent: bool = field(default=True, init=False)
    _blind: int = field(default=0, init=False)
    _last_px: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(self.layout.dictionary)
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def update(self, frame: np.ndarray) -> Calibration:
        due = self._urgent or self._frame % self.refresh_every == 0
        self._frame += 1
        if due:
            self.current = self.refresh(frame)
        return self.current

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """Marker id -> its four corners in pixels, for the ids the layout names."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return {}
        return {
            int(i): c.reshape(4, 2).astype(np.float64)
            for i, c in zip(ids.ravel(), corners, strict=True)
            if int(i) in self.layout.centres
        }

    def refresh(self, frame: np.ndarray) -> Calibration:
        found = self.detect(frame)
        centres = {i: c.mean(axis=0) for i, c in found.items()}
        self._urgent = False

        if len(centres) == len(self.layout.centres):
            moved = self._moved(centres)
            ids = sorted(centres)
            h = fit([tuple(centres[i]) for i in ids], [self.layout.centres[i] for i in ids])
            self._last_px = centres
            self._blind = 0
            # A move means the next frame may still be mid-shake: look again.
            self._urgent = moved
            return Calibration("ok", h, frozenset(centres), self._side_error(h, found))

        previous = self.current
        if previous.homography is None:
            return Calibration("uncalibrated", None, frozenset(centres))

        if centres:
            self._blind = 0
            if self._moved(centres):
                # The camera moved and we cannot refit: say so.
                self._urgent = True
                return Calibration("lost", None, frozenset(centres))
            return Calibration("held", previous.homography, frozenset(centres),
                               previous.error_studs)

        self._blind += 1
        if self._blind > self.grace:
            return Calibration("lost", None)
        return Calibration("stale", previous.homography, frozenset(), previous.error_studs)

    def _moved(self, centres: Mapping[int, np.ndarray]) -> bool:
        return any(
            i in self._last_px and float(np.linalg.norm(c - self._last_px[i])) > self.shift_px
            for i, c in centres.items()
        )

    def _side_error(self, h: np.ndarray, found: Mapping[int, np.ndarray]) -> float:
        worst = 0.0
        for corners in found.values():
            studs = _map(h, corners)
            sides = np.linalg.norm(studs - np.roll(studs, -1, axis=0), axis=1)
            worst = max(worst, float(np.max(np.abs(sides - self.layout.side_studs))))
        return worst
