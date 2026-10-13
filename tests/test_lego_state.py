"""Stud geometry and the build-state tracker (FR-STA-2..4, FR-STA-6)."""

from __future__ import annotations

import pytest

from thirdeye.compiler.schema import PartRef, Pose
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import (
    Reading,
    StudGrid,
    apply_homography,
    extent,
    pose_matches,
    rotation_from_extent,
    rotations_indistinguishable,
    stud_distance,
)

RED = PartRef(color="red", size="2x4")
BLUE = PartRef(color="blue", size="2x2")
YELLOW = PartRef(color="yellow", size="1x4")


@pytest.fixture
def grid() -> StudGrid:
    return StudGrid(16, 16)


def reading(part: PartRef, x: int, y: int, **kw) -> Reading:
    return Reading(part=part, pose=Pose(x=x, y=y, **kw.pop("pose", {})), residual=0.0, **kw)


# --- the homography -----------------------------------------------------------------

def test_a_scale_and_offset_maps_pixels_to_studs() -> None:
    """32 px per stud with the plate origin at pixel (100, 50)."""
    h = [[1 / 32, 0, -100 / 32], [0, 1 / 32, -50 / 32], [0, 0, 1]]
    assert apply_homography(h, 100, 50) == pytest.approx((0.0, 0.0))
    assert apply_homography(h, 228, 114) == pytest.approx((4.0, 2.0))


def test_a_perspective_term_divides_through() -> None:
    h = [[2, 0, 0], [0, 2, 0], [0, 0, 4]]
    assert apply_homography(h, 8, 4) == pytest.approx((4.0, 2.0))


def test_a_degenerate_point_is_refused_not_silently_infinite() -> None:
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="degenerate"):
        apply_homography(h, 1, 1)


# --- extents and rotation ----------------------------------------------------------

@pytest.mark.parametrize(
    ("part", "rot", "expected"),
    [(RED, 0, (2, 4)), (RED, 90, (4, 2)), (RED, 180, (2, 4)), (RED, 270, (4, 2)),
     (BLUE, 90, (2, 2))],
)
def test_extent_swaps_on_a_quarter_turn(part, rot, expected) -> None:
    assert extent(part, rot) == expected


@pytest.mark.parametrize(
    ("observed", "expected"),
    [((2.1, 4.0), 0), ((4.0, 2.1), 90), ((1.9, 3.8), 0), ((3.9, 2.2), 90)],
)
def test_rotation_is_recovered_from_the_box(observed, expected) -> None:
    assert rotation_from_extent(*observed, RED) == expected


def test_a_square_part_has_no_recoverable_rotation() -> None:
    assert rotation_from_extent(2.0, 2.0, BLUE) == 0
    assert rotations_indistinguishable(BLUE, 0, 270)


def test_pose_matches_ignores_only_unobservable_rotations() -> None:
    assert pose_matches(Pose(x=0, y=0, rot=180), Pose(x=0, y=0, rot=0), RED)
    assert not pose_matches(Pose(x=0, y=0, rot=90), Pose(x=0, y=0, rot=0), RED)
    assert not pose_matches(Pose(x=1, y=0), Pose(x=0, y=0), RED)
    assert not pose_matches(Pose(x=0, y=0, layer=1), Pose(x=0, y=0), RED)
    assert stud_distance(Pose(x=1, y=2, layer=1), Pose(x=0, y=0)) == 4


# --- snapping a detection onto the grid --------------------------------------------

def test_a_centroid_snaps_to_the_low_corner(grid: StudGrid) -> None:
    """A 2x4 centred at (1, 2) has its low corner at the origin."""
    got = grid.read(RED, (1.0, 2.0))
    assert got is not None
    assert (got.pose.x, got.pose.y) == (0, 0)
    assert got.residual == pytest.approx(0.0)


def test_the_residual_is_the_offset_the_evaluation_reports(grid: StudGrid) -> None:
    got = grid.read(RED, (1.3, 2.0))
    assert got is not None
    assert (got.pose.x, got.pose.y) == (0, 0)
    assert got.residual == pytest.approx(0.3)


def test_a_rotated_part_snaps_against_its_rotated_extent(grid: StudGrid) -> None:
    got = grid.read(RED, (2.0, 1.0), rot=90)
    assert got is not None
    assert (got.pose.x, got.pose.y, got.pose.rot) == (0, 0, 90)


@pytest.mark.parametrize("centroid", [(-2.0, 2.0), (1.0, -2.0), (16.0, 2.0), (1.0, 15.0)])
def test_a_brick_off_the_plate_is_ignored_not_an_error(grid: StudGrid, centroid) -> None:
    """Bricks in the box beside the plate are a normal thing to detect."""
    assert grid.read(RED, centroid) is None


def test_fits_checks_the_far_corner(grid: StudGrid) -> None:
    assert grid.fits(RED, Pose(x=14, y=12))
    assert not grid.fits(RED, Pose(x=15, y=12))
    assert grid.fits(RED, Pose(x=12, y=14, rot=90))


# --- the stability filter (FR-STA-3) -----------------------------------------------

def test_a_brick_is_placed_only_after_it_has_held_still() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)])
    assert state.diff() == []        # seen once, not yet settled
    state.observe(0.3, [reading(RED, 0, 0)])
    assert state.diff() == []
    state.observe(0.6, [reading(RED, 0, 0)])
    events = state.diff()
    assert [e.kind for e in events] == ["ADD"]
    assert events[0].part == RED
    assert state.current()[0].part == RED


def test_a_flicker_never_becomes_a_placement() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    for t in (0.0, 0.1, 0.2):
        state.observe(t, [reading(RED, 0, 0)])
        state.observe(t + 0.05, [])
    assert state.diff() == []
    assert state.current() == ()


def test_a_momentary_miss_does_not_remove_a_placed_brick() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)])
    state.observe(1.0, [reading(RED, 0, 0)])
    assert [e.kind for e in state.diff()] == ["ADD"]

    state.observe(1.1, [])                       # one bad frame
    state.observe(1.2, [reading(RED, 0, 0)])
    assert state.diff() == []
    assert len(state.current()) == 1


def test_a_brick_genuinely_taken_off_is_removed() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)])
    state.observe(1.0, [reading(RED, 0, 0)])
    state.diff()

    state.observe(1.1, [])
    state.observe(2.0, [])
    events = state.diff()
    assert [e.kind for e in events] == ["REMOVE"]
    assert events[0].pose == Pose(x=0, y=0)
    assert state.current() == ()


def test_swapping_the_brick_in_a_cell_reports_both_halves() -> None:
    """The checker needs the removal to reopen the step before it judges the
    replacement, or the new brick looks like it matched nothing."""
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(BLUE, 0, 0)])
    state.observe(1.0, [reading(BLUE, 0, 0)])
    state.diff()

    state.observe(2.0, [reading(YELLOW, 0, 0)])
    state.observe(3.0, [reading(YELLOW, 0, 0)])
    assert [e.kind for e in state.diff()] == ["REMOVE", "ADD"]


def test_a_part_changing_identity_restarts_its_timer() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(BLUE, 0, 0)])
    state.observe(0.4, [reading(YELLOW, 0, 0)])   # the detector changed its mind
    state.observe(0.6, [reading(YELLOW, 0, 0)])
    assert state.diff() == []                     # only 0.2 s of agreement
    state.observe(1.0, [reading(YELLOW, 0, 0)])
    assert [e.kind for e in state.diff()] == ["ADD"]


# --- freezing under occlusion (FR-PER-6) ------------------------------------------

def test_nothing_commits_while_a_hand_is_over_the_plate() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    for t in (0.0, 0.5, 1.0, 1.5):
        state.observe(t, [reading(RED, 0, 0)], hands_present=True)
    assert state.diff() == []
    assert state.occluded


def test_the_timer_restarts_when_the_hand_leaves() -> None:
    """A brick must not be credited for time it spent behind a hand: what was
    behind the hand is exactly what we could not see."""
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)], hands_present=True)
    state.observe(1.0, [reading(RED, 0, 0)], hands_present=True)
    state.observe(1.1, [reading(RED, 0, 0)])
    assert state.diff() == []
    state.observe(1.7, [reading(RED, 0, 0)])
    assert [e.kind for e in state.diff()] == ["ADD"]


def test_a_placed_brick_survives_being_hidden() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)])
    state.observe(1.0, [reading(RED, 0, 0)])
    state.diff()
    for t in (1.1, 2.0, 3.0):
        state.observe(t, [], hands_present=True)
    assert state.diff() == []
    assert len(state.current()) == 1


# --- dropping what is not usable ---------------------------------------------------

def test_unconfident_and_off_grid_detections_are_dropped_and_counted() -> None:
    state = BuildState(StudGrid(16, 16), min_confidence=0.5, max_residual=0.5)
    state.observe(0.0, [
        Reading(part=RED, pose=Pose(x=0, y=0), residual=0.0, confidence=0.2),
        Reading(part=BLUE, pose=Pose(x=4, y=0), residual=0.9, confidence=1.0),
    ])
    state.observe(1.0, [])
    assert state.diff() == []
    assert state.dropped == 2


# --- re-verification (FR-STA-6) ---------------------------------------------------

def test_a_re_read_waits_for_a_clear_view() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.request_reverify({"s1"})
    assert state.pending_reverify() == {"s1"}

    state.observe(0.1, [reading(RED, 0, 0)], hands_present=True)
    assert state.reverify_ready() is False
    assert state.take_reverification({"s1"}) is None

    state.observe(0.2, [reading(RED, 0, 0)])
    assert state.reverify_ready() is True
    rev = state.take_reverification({"s1"})
    assert rev is not None and rev.supported == {"s1"}
    assert state.pending_reverify() == set()


def test_confirmed_at_answers_the_question_the_checker_asks() -> None:
    state = BuildState(StudGrid(16, 16), stable_s=0.5)
    state.observe(0.0, [reading(RED, 0, 0)])
    state.observe(1.0, [reading(RED, 0, 0)])
    assert state.confirmed_at(RED, Pose(x=0, y=0)) is True
    assert state.confirmed_at(RED, Pose(x=0, y=0, rot=180)) is True   # unobservable
    assert state.confirmed_at(BLUE, Pose(x=0, y=0)) is False
    assert state.confirmed_at(RED, Pose(x=4, y=0)) is False


def test_a_stud_grid_offers_no_lattice_predicates() -> None:
    """So a Jenga rulebook is refused at setup rather than mid-session."""
    assert BuildState(StudGrid(16, 16)).predicates() == ({}, {})
