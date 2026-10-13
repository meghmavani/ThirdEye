"""Top-down calibration against synthetic scenes (FR-CAL-1..4, FR-CAL-7).

See `synthetic.py`: the camera's perspective is known, so the fit is checked
against the true pixel -> stud map.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from synthetic import LAYOUT, scene, to_pixels

from thirdeye.perception.calibrate import Calibrator, fit, from_corners
from thirdeye.state.lego.grid import apply_homography


def test_markers_fit_a_homography_onto_studs_from_an_angle() -> None:
    cal = Calibrator(LAYOUT).update(scene())
    assert cal.status == "ok" and cal.usable
    assert cal.seen == frozenset({0, 1, 2, 3})
    for stud in [(0.0, 0.0), (16.0, 16.0), (3.5, 11.25), (8.0, 0.5)]:
        got = apply_homography(cal.homography.tolist(), *to_pixels(stud))
        assert got == pytest.approx(stud, abs=0.1)


def test_the_fit_reports_how_far_to_trust_it() -> None:
    cal = Calibrator(LAYOUT).update(scene())
    assert cal.error_studs is not None and cal.error_studs < 0.2


def test_a_hand_over_one_marker_keeps_the_fit() -> None:
    calib = Calibrator(LAYOUT, refresh_every=1)
    first = calib.update(scene())
    held = calib.update(scene(hidden=frozenset({2})))
    assert held.status == "held" and held.usable
    assert np.allclose(held.homography, first.homography)


def test_a_camera_knocked_while_a_marker_is_hidden_loses_calibration() -> None:
    calib = Calibrator(LAYOUT, refresh_every=1)
    calib.update(scene())
    lost = calib.update(scene(hidden=frozenset({2}), shift=(25, 0)))
    assert lost.status == "lost"
    assert not lost.usable and lost.homography is None


def test_a_knocked_camera_with_all_markers_visible_is_refitted_at_once() -> None:
    calib = Calibrator(LAYOUT, refresh_every=10)
    calib.update(scene())
    for _ in range(9):
        calib.update(scene())
    moved = calib.update(scene(shift=(25, 0)))     # the 10th: a scheduled refresh
    assert moved.status == "ok"
    # ...and the very next frame is looked at again, not ten frames later.
    assert calib._urgent


def test_nothing_in_view_is_tolerated_briefly_then_reported() -> None:
    calib = Calibrator(LAYOUT, refresh_every=1, grace=2)
    calib.update(scene())
    blank = np.full_like(scene(), 255)
    assert [calib.update(blank).status for _ in range(3)] == ["stale", "stale", "lost"]


def test_no_markers_ever_means_uncalibrated_not_a_guess() -> None:
    cal = Calibrator(LAYOUT).update(np.full((300, 300), 255, np.uint8))
    assert cal.status == "uncalibrated" and not cal.usable


def test_detection_runs_only_on_the_refresh_cadence() -> None:
    calib = Calibrator(LAYOUT, refresh_every=10)
    calib.update(scene())
    blank = np.full_like(scene(), 255)
    # Nine frames of nothing between refreshes are never even looked at.
    assert all(calib.update(blank).status == "ok" for _ in range(9))
    assert calib.update(blank).status == "stale"


def test_grey_frames_are_accepted() -> None:
    gray = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    assert Calibrator(LAYOUT).update(gray).status == "ok"


def test_four_clicked_corners_are_the_fallback() -> None:
    corners = [to_pixels(c) for c in [(0, 0), (16, 0), (16, 16), (0, 16)]]
    cal = from_corners(corners, 16, 16)
    got = apply_homography(cal.homography.tolist(), *to_pixels((5.0, 7.0)))
    assert got == pytest.approx((5.0, 7.0), abs=1e-6)


def test_degenerate_points_are_refused() -> None:
    with pytest.raises(ValueError):
        fit([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        fit([(0, 0), (1, 1)], [(0, 0), (1, 0)])
