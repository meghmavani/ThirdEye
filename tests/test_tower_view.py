"""Jenga slot reading from depth, diagonal-corner mode (FR-DEP-1..4, FR-CAL-8, FR-STA-7)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from tower_synthetic import (
    K,
    corner_camera,
    full,
    render,
    with_block,
    without,
)

from thirdeye.events import Slot
from thirdeye.perception.camera_model import PinholeCamera, PoseError, TableMarkers, pose_from_markers
from thirdeye.perception.tower_view import (
    SlotReader,
    Thresholds,
    TowerReader,
    slot_windows,
    tower_region_px,
)

CAM1 = corner_camera(-1, -1)
CAM2 = corner_camera(+1, +1)


def truth(occ: list[list[int]], layers: int = 22) -> dict[tuple[int, int], str]:
    return {
        (l, s): "occupied" if l < len(occ) and occ[l][s] else "empty"
        for l in range(layers) for s in range(3)
    }


def states(reading) -> dict[tuple[int, int], str]:
    return {k: o.state for k, o in reading.obs.items()}


def accuracy(reading, occ) -> float:
    want = truth(occ)
    got = states(reading)
    return sum(got[k] == v for k, v in want.items()) / len(want)


# --- camera model --------------------------------------------------------------------

def test_the_tower_centre_projects_to_the_image_centre_and_rays_round_trip() -> None:
    px, z = CAM1.project(np.array([[37.5, 37.5, 160.0]]))
    assert abs(px[0, 0] - 128) < 1e-6 and abs(px[0, 1] - 96) < 1e-6 and abs(z[0] - 520) < 1e-6
    ray = CAM1.rays()[96, 128]
    assert np.allclose(CAM1.c + ray * 520, [37.5, 37.5, 160.0], atol=1e-6)


def test_each_corner_camera_faces_two_tower_faces() -> None:
    assert CAM1.view_cosine(np.array([0, 37.5, 100]), np.array([-1.0, 0, 0])) > 0.6
    assert CAM1.view_cosine(np.array([37.5, 0, 100]), np.array([0, -1.0, 0])) > 0.6
    assert CAM1.view_cosine(np.array([75, 37.5, 100]), np.array([1.0, 0, 0])) < 0


def _marker_scene(camera: PinholeCamera, markers: TableMarkers, hidden=()):
    """A colour image of the table markers as `camera` sees them."""
    img = np.full((camera.height, camera.width, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(markers.dictionary)
    side = 200
    src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    for mid in markers.centres:
        if mid in hidden:
            continue
        px, _ = camera.project(markers.corners(mid))
        m = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, mid, side), cv2.COLOR_GRAY2BGR)
        h = cv2.getPerspectiveTransform(src, px.astype(np.float32))
        warped = cv2.warpPerspective(m, h, (camera.width, camera.height), borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), h,
                                   (camera.width, camera.height)) > 0
        img[mask] = warped[mask]
    return img


def big_camera(sign: int):
    """A higher-resolution colour camera, as the phone's RGB stream is."""
    k = np.array([[1080.0, 0, 640], [0, 1080.0, 360], [0, 0, 1]])
    d = 520 / np.sqrt(2)
    return PinholeCamera.look_at((37.5 + sign * d, 37.5 + sign * d, 260.0), (37.5, 37.5, 0.0),
                                 k, (1280, 720))


def test_the_camera_pose_is_recovered_from_the_table_markers() -> None:
    markers = TableMarkers.around_tower()
    cam = big_camera(-1)
    est = pose_from_markers(_marker_scene(cam, markers), cam.k, markers)
    assert np.linalg.norm(est.c - cam.c) < 8.0  # millimetres, at half a metre
    pts = np.array([[0.0, 0.0, 0.0], [75.0, 75.0, 270.0]])
    assert np.abs(est.project(pts)[0] - cam.project(pts)[0]).max() < 2.0


def test_too_few_markers_refuses_a_pose() -> None:
    markers = TableMarkers.around_tower()
    cam = big_camera(-1)
    with pytest.raises(PoseError):
        pose_from_markers(_marker_scene(cam, markers, hidden=(0, 1, 2)), cam.k, markers)


# --- windows ---------------------------------------------------------------------------

def test_every_slot_has_two_end_on_windows_and_even_odd_layers_use_different_faces() -> None:
    wins = slot_windows(18)
    assert len(wins) == 18 * 3 * 2
    even = [w for w in wins if w.layer == 0]
    odd = [w for w in wins if w.layer == 1]
    assert all(abs(w.normal[0]) == 1 for w in even) and all(abs(w.normal[1]) == 1 for w in odd)


def test_a_corner_camera_sees_every_slot_end_on_without_the_other_camera() -> None:
    reader = SlotReader([CAM1], layers=22)
    seen = {(w.layer, w.slot) for w in reader.plans[0].windows}
    assert len(seen) == 22 * 3


# --- reading ---------------------------------------------------------------------------

def test_a_full_tower_reads_full_and_the_empty_layers_above_read_empty() -> None:
    occ = full()
    reader = SlotReader([CAM1, CAM2], layers=22)
    reading = reader.read([render(CAM1, occ), render(CAM2, occ)])
    assert accuracy(reading, occ) == 1.0


def test_one_corner_alone_reads_every_slot_correctly() -> None:
    rng = np.random.default_rng(1)
    occ = full()
    for _ in range(14):
        occ[int(rng.integers(0, 17))][int(rng.integers(0, 3))] = 0
    reader = SlotReader([CAM1], layers=22)
    assert accuracy(reader.read([render(CAM1, occ)]), occ) == 1.0
    reader2 = SlotReader([CAM2], layers=22)
    assert accuracy(reader2.read([render(CAM2, occ)]), occ) == 1.0


def test_two_cameras_merged_stay_correct_through_noise_and_dropouts() -> None:
    rng = np.random.default_rng(2)
    wrong = total = 0
    reader = SlotReader([CAM1, CAM2], layers=22)
    for trial in range(4):
        occ = full()
        for _ in range(10):
            occ[int(rng.integers(0, 17))][int(rng.integers(0, 3))] = 0
        occ = with_block(occ, (18, 0), (18, 1))
        frames = [render(c, occ, noise_mm=2.5, dropout=0.02, seed=trial) for c in (CAM1, CAM2)]
        reading = reader.read(frames)
        want = truth(occ)
        for k, v in states(reading).items():
            total += 1
            if reading.obs[k].state != "unobserved":
                wrong += v != want[k]
    assert wrong == 0, f"{wrong} of {total} slots misread"


def test_a_block_pushed_out_is_still_occupied_and_is_flagged_pushed() -> None:
    occ = full()
    reader = SlotReader([CAM1], layers=22)
    reading = reader.read([render(CAM1, occ, pushed={(6, 1): 14.0})])
    o = reading.at(6, 1)
    assert o.state == "occupied" and o.pushed_mm > 8
    assert reading.at(6, 0).pushed_mm == 0 and reading.at(7, 1).pushed_mm == 0


def test_a_hand_in_front_makes_slots_unobserved_not_empty() -> None:
    occ = full()
    hand = (-120.0, -120.0, 60.0, -40.0, -40.0, 120.0)  # a box between camera 1 and the tower
    reader = SlotReader([CAM1], layers=22)
    reading = reader.read([render(CAM1, occ, hand=hand)])
    hidden = [k for k, o in reading.obs.items() if o.state == "unobserved"]
    assert hidden, "a hand across the face should hide some slots"
    assert all(o.state != "empty" for k, o in reading.obs.items() if k[0] < 18)


def test_a_dropped_camera_frame_just_leaves_the_other_camera() -> None:
    occ = without(full(), (4, 1))
    reader = SlotReader([CAM1, CAM2], layers=22)
    reading = reader.read([None, render(CAM2, occ)])
    assert accuracy(reading, occ) == 1.0


def test_two_cameras_that_disagree_give_unobserved_never_a_guess() -> None:
    a, b = full(), without(full(), (5, 0))
    reader = SlotReader([CAM1, CAM2], layers=22)
    reading = reader.read([render(CAM1, a), render(CAM2, b)])
    o = reading.at(5, 0)
    assert o.state == "unobserved" and o.contradictory


def test_slots_no_camera_can_see_are_unobserved_and_the_belief_holds() -> None:
    occ = full()
    reader = TowerReader([CAM1], thresholds=Thresholds(min_cos=0.95))  # nothing is face-on enough
    update = reader.update([render(CAM1, without(occ, (3, 0)))])
    assert len(update.unobserved) == 22 * 3
    assert update.snapshot == tuple(tuple(r) for r in occ)  # belief untouched, no invented pull


def test_the_colour_cue_resolves_a_residual_too_small_to_call_and_fails_safe_without_it() -> None:
    occ = without(full(), (8, 1))
    wide = Thresholds(empty=45.0)  # the tunnel's residual now falls in the ambiguous band
    frame = render(CAM1, occ)
    with_cue = SlotReader([CAM1], layers=22, thresholds=wide, use_colour=True).read([frame])
    without_cue = SlotReader([CAM1], layers=22, thresholds=wide, use_colour=False).read([frame])
    assert with_cue.at(8, 1).state == "empty"
    assert without_cue.at(8, 1).state == "unobserved"


# --- the belief ------------------------------------------------------------------------

def test_the_reader_reports_a_pull_then_a_placement_on_a_new_layer() -> None:
    reader = TowerReader([CAM1, CAM2])
    first = reader.update([render(CAM1, full()), render(CAM2, full())])
    assert len(first.snapshot) == 18 and first.height_mm == 18 * 15
    pulled = without(full(), (5, 1))
    second = reader.update([render(CAM1, pulled), render(CAM2, pulled)])
    assert second.snapshot[5] == (1, 0, 1)
    placed = with_block(pulled, (18, 0))
    third = reader.update([render(CAM1, placed), render(CAM2, placed)])
    assert len(third.snapshot) == 19 and third.snapshot[18] == (1, 0, 0)
    assert third.height_mm == 19 * 15


def test_a_collapse_reads_as_a_height_far_below_the_belief() -> None:
    reader = TowerReader([CAM1, CAM2])
    reader.update([render(CAM1, full()), render(CAM2, full())])
    rubble = [[0, 0, 0]]
    update = reader.update([render(CAM1, rubble), render(CAM2, rubble)])
    assert update.height_mm <= 15


def test_a_hand_across_the_upper_layers_does_not_look_like_a_collapse() -> None:
    reader = TowerReader([CAM1])
    reader.update([render(CAM1, full())])
    hand = (-150.0, -150.0, 150.0, -30.0, -30.0, 300.0)
    update = reader.update([render(CAM1, full(), hand=hand)])
    assert update.height_mm >= 17 * 15


def test_slot_reading_for_two_cameras_is_fast_enough() -> None:
    reader = SlotReader([CAM1, CAM2], layers=22)
    frames = [render(CAM1, full()), render(CAM2, full())]
    reader.read(frames)  # warm up
    start = time.perf_counter()
    for _ in range(10):
        reader.read(frames)
    per = (time.perf_counter() - start) / 10 * 1000
    assert per < 15.0 * 3, f"{per:.1f} ms per tick"  # NFR-PERF-3 is 15 ms; slack for CI


def test_the_tower_outline_for_hand_counting_contains_the_tower() -> None:
    region = tower_region_px(CAM1)
    px, _ = CAM1.project(np.array([[37.5, 37.5, 135.0]]))
    assert cv2.pointPolygonTest(region.reshape(-1, 1, 2), (float(px[0, 0]), float(px[0, 1])), False) >= 0
    assert isinstance(Slot(0, 0), Slot) and K.shape == (3, 3)
