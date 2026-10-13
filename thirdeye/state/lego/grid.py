"""Stud-grid geometry (FR-CAL-*, FR-STA-2).

Everything about turning what the camera saw into stud coordinates, and about
what two poses on a stud grid mean relative to each other. Deliberately has no
dependency on OpenCV: `perception/calibrate.py` produces the homography, this
module only applies it, so the geometry is testable with a matrix literal and no
camera.

Stud coordinates are continuous here and integral in a `Pose`. The gap between
them is the `residual`, which is the detector metric plan section 7 actually
cares about -- median centroid error <= 0.4 stud, p95 <= 1.0 -- because mAP@0.5
on a 2x4 brick allows roughly a full stud of slack and a "passing" detection can
still trip a wrong-position alert.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from thirdeye.compiler.schema import PartRef, Pose, Rotation

#: A 3x3 pixel -> stud homography. A nested sequence or a NumPy array both work;
#: taking it structurally keeps this module free of a NumPy import.
Homography = Sequence[Sequence[float]]

#: LEGO geometry, for the side-camera height work and for reporting in real units.
STUD_PITCH_MM = 8.0
BRICK_HEIGHT_MM = 9.6
PLATE_HEIGHT_MM = 3.2


def apply_homography(h: Homography, px: float, py: float) -> tuple[float, float]:
    """Map one image point to continuous stud coordinates."""
    u = h[0][0] * px + h[0][1] * py + h[0][2]
    v = h[1][0] * px + h[1][1] * py + h[1][2]
    w = h[2][0] * px + h[2][1] * py + h[2][2]
    if abs(w) < 1e-9:
        raise ValueError("point maps to the horizon; the homography is degenerate here")
    return u / w, v / w


def rotations_indistinguishable(part: PartRef, a: int, b: int) -> bool:
    """Would these two rotations of this part look identical on the plate?

    A 2x4 brick at 0 and at 180 degrees covers the same studs and presents the
    same face; a square part is the same at all four. The checker must not alert
    on a difference no camera can see -- that would be a false alert by
    construction, against a budget of 1.0 per build (NFR-REL-2).
    """
    w, h = part.dims
    if w == h:
        return True
    return a % 180 == b % 180


def pose_matches(observed: Pose, expected: Pose, part: PartRef) -> bool:
    """Same cell, same layer, and a rotation we could not tell apart anyway."""
    return (
        observed.x == expected.x
        and observed.y == expected.y
        and observed.layer == expected.layer
        and rotations_indistinguishable(part, observed.rot, expected.rot)
    )


def same_cell(a: Pose, b: Pose) -> bool:
    return a.x == b.x and a.y == b.y and a.layer == b.layer


def stud_distance(a: Pose, b: Pose) -> int:
    """Manhattan distance in studs, layers included."""
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.layer - b.layer)


def extent(part: PartRef, rot: int) -> tuple[int, int]:
    """How many studs the part spans in x and y at this rotation."""
    w, h = part.dims
    return (h, w) if rot % 180 == 90 else (w, h)


def rotation_from_extent(observed_x: float, observed_y: float, part: PartRef) -> Rotation:
    """Which rotation the observed bounding box implies.

    Top-down we can only recover the axis, never the direction: 0 and 180 are
    indistinguishable, so this returns the representative of the pair. That is
    not a limitation to work around -- `rotations_indistinguishable` means the
    checker never asks for more.
    """
    w, h = part.dims
    if w == h:
        return 0
    return 0 if (observed_y >= observed_x) == (h > w) else 90


@dataclass(frozen=True)
class Reading:
    """One part the detector found, resolved onto the grid.

    `residual` is how far the observed centroid sits from the centre of the cell
    it was snapped to, in studs. The tracker uses it to reject a detection that
    is not really on the grid, and the evaluation reports its median and p95.
    """

    part: PartRef
    pose: Pose
    residual: float
    confidence: float = 1.0

    @property
    def cell(self) -> tuple[int, int, int]:
        return (self.pose.x, self.pose.y, self.pose.layer)

    def __str__(self) -> str:
        return f"{self.part} @ ({self.pose.x},{self.pose.y},L{self.pose.layer},rot{self.pose.rot})"


class StudGrid:
    """The baseplate, in studs."""

    def __init__(self, studs_x: int, studs_y: int):
        self.studs_x = studs_x
        self.studs_y = studs_y

    @classmethod
    def from_spec(cls, baseplate) -> StudGrid:
        return cls(baseplate.studs_x, baseplate.studs_y)

    def contains(self, x: int, y: int) -> bool:
        return 0 <= x < self.studs_x and 0 <= y < self.studs_y

    def fits(self, part: PartRef, pose: Pose) -> bool:
        w, h = extent(part, pose.rot)
        return self.contains(pose.x, pose.y) and self.contains(pose.x + w - 1, pose.y + h - 1)

    def read(
        self,
        part: PartRef,
        centroid: tuple[float, float],
        rot: Rotation = 0,
        layer: int = 0,
        confidence: float = 1.0,
    ) -> Reading | None:
        """Snap a detected centroid to a cell, or None if it is not on the plate.

        The centroid is the middle of the part's footprint, so the low corner is
        half the extent away. Returning None rather than raising is deliberate:
        a brick in the box beside the plate is a normal thing for the detector to
        see, not an error, and the tracker simply ignores it.
        """
        cx, cy = centroid
        w, h = extent(part, rot)
        x = round(cx - w / 2)
        y = round(cy - h / 2)
        if x < 0 or y < 0:
            return None
        pose = Pose(x=x, y=y, layer=layer, rot=rot)
        if not self.fits(part, pose):
            return None
        residual = max(abs(cx - (x + w / 2)), abs(cy - (y + h / 2)))
        return Reading(part=part, pose=pose, residual=residual, confidence=confidence)
