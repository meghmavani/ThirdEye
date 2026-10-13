"""From pixels to grid readings (FR-PER-1..4, FR-PER-6), on synthetic scenes.

The detector is replaced by a perfect one that returns each brick's true box,
so these tests isolate everything *after* detection: the homography, size and
rotation from stud extents, colour from hue, the hand freeze, and the
calibration gate. The real models are exercised in `test_models.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import LAYOUT, STUDS, brick_box, scene, to_pixels

from thirdeye.compiler.schema import PartRef, Pose
from thirdeye.perception.calibrate import Calibrator
from thirdeye.perception.detect import (
    Detection,
    dominant_color,
    part_from_class,
    size_from_extent,
    to_readings,
)
from thirdeye.perception.hands import Hand, hands_in_contact, is_grasping, over_workspace
from thirdeye.perception.pipeline import TopDownPipeline
from thirdeye.state.lego.grid import StudGrid

GRID = StudGrid(STUDS, STUDS)
BRICKS = [
    (PartRef(color="red", size="2x4"), Pose(x=0, y=0, rot=90)),
    (PartRef(color="blue", size="2x2"), Pose(x=6, y=2)),
    (PartRef(color="yellow", size="1x4"), Pose(x=10, y=9)),
    (PartRef(color="green", size="2x4"), Pose(x=12, y=4)),
]


class Perfect:
    """A detector that finds exactly the drawn bricks, naming none of them."""

    def __init__(self, bricks, named: bool = False):
        self.bricks, self.named = bricks, named

    def detect(self, frame):
        return [Detection(brick_box(p, q), 0.9, part=p if self.named else None)
                for p, q in self.bricks]


class FakeHands:
    def __init__(self, hands):
        self.hands = hands

    def detect(self, frame, t):
        return self.hands


def _hand_at(stud: tuple[float, float], spread_px: float = 30.0) -> Hand:
    """A hand-shaped cloud of 21 keypoints centred over a stud point."""
    cx, cy = to_pixels(stud)
    rng = np.random.default_rng(0)
    return Hand(np.array([cx, cy]) + rng.normal(0, spread_px, (21, 2)))


def _pipeline(bricks=BRICKS, hands=None, named=False) -> TopDownPipeline:
    return TopDownPipeline(Calibrator(LAYOUT), Perfect(bricks, named), GRID,
                           hands=FakeHands(hands or []))


@pytest.mark.parametrize("named", [False, True])
def test_every_brick_lands_on_its_cell(named: bool) -> None:
    result = _pipeline(named=named).process(scene(bricks=BRICKS), t=0.0)
    assert result.usable and not result.hands_present
    got = {(str(r.part), r.pose.x, r.pose.y, r.pose.rot % 180) for r in result.readings}
    want = {(str(p), q.x, q.y, q.rot % 180) for p, q in BRICKS}
    assert got == want
    assert max(r.residual for r in result.readings) < 0.2


def test_a_hand_over_the_plate_freezes_the_frame() -> None:
    result = _pipeline(hands=[_hand_at((8, 8))]).process(scene(bricks=BRICKS), t=0.0)
    assert result.hands_present
    assert result.readings == []           # the detector was not even run


def test_a_hand_beside_the_plate_does_not() -> None:
    result = _pipeline(hands=[_hand_at((30, 8), spread_px=5)]).process(scene(bricks=BRICKS), 0.0)
    assert not result.hands_present
    assert len(result.readings) == len(BRICKS)


def test_no_calibration_means_no_readings() -> None:
    blank = np.full_like(scene(), 255)
    result = _pipeline().process(blank, t=0.0)
    assert not result.usable and result.readings == []


def test_the_pipeline_times_every_stage() -> None:
    pipe = _pipeline()
    for i in range(3):
        pipe.process(scene(bricks=BRICKS), t=i / 30)
    table = pipe.timings.table()
    for stage in ("calibrate", "hands", "detect", "to_grid", "total"):
        assert stage in table


# --- the pieces ------------------------------------------------------------------

@pytest.mark.parametrize(("ex", "ey", "size", "rot"), [
    (2.1, 3.9, "2x4", 0), (3.8, 2.2, "2x4", 90), (2.0, 2.0, "2x2", 0),
    (0.9, 4.1, "1x4", 0), (0.2, 0.3, "1x1", 0),
])
def test_size_and_rotation_come_from_stud_extent(ex, ey, size, rot) -> None:
    assert size_from_extent(ex, ey) == (size, rot)


@pytest.mark.parametrize("color", ["red", "blue", "yellow", "green", "white", "black"])
def test_brick_colour_is_read_from_hue(color: str) -> None:
    from synthetic import BGR

    patch = np.zeros((40, 40, 3), np.uint8)
    patch[:] = BGR[color]
    assert dominant_color(patch, (0, 0, 40, 40)) == color


def test_class_names_are_part_names() -> None:
    assert part_from_class("red_2x4") == PartRef(color="red", size="2x4")
    assert part_from_class("hand") is None


def test_off_plate_detections_are_dropped() -> None:
    far = Detection(box=(0, 0, 10, 10), confidence=0.9, part=PartRef(color="red", size="2x2"))
    cal = Calibrator(LAYOUT).update(scene())
    assert to_readings([far], scene(), cal.homography.tolist(), GRID) == []


def test_grasp_is_a_pinch_relative_to_hand_size() -> None:
    kp = np.zeros((21, 2))
    kp[9] = (0, 100)                   # middle knuckle: hand size 100 px
    kp[4], kp[8] = (10, 50), (20, 50)  # thumb and index 10 px apart
    assert is_grasping(Hand(kp))
    kp[8] = (80, 50)
    assert not is_grasping(Hand(kp))


def test_hands_touching_a_region_are_counted() -> None:
    tower = np.array([[100, 100], [200, 100], [200, 400], [100, 400]])
    touching = Hand(np.full((21, 2), 150.0))
    away = Hand(np.full((21, 2), 500.0))
    assert hands_in_contact([touching, away], tower) == 1
    assert hands_in_contact([touching, touching], tower) == 2


def test_over_workspace_uses_the_calibration() -> None:
    h = Calibrator(LAYOUT).update(scene()).homography.tolist()
    assert over_workspace(_hand_at((8, 8), spread_px=3), h, STUDS, STUDS)
    assert not over_workspace(_hand_at((-6, -6), spread_px=3), h, STUDS, STUDS)
