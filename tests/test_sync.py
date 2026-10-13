"""Time alignment between the Jenga cameras (FR-CAL-10)."""

from __future__ import annotations

import numpy as np

from thirdeye.perception.sync import estimate_offset, motion_signal, pair_frames


def bursts(n: int, at: list[int]) -> np.ndarray:
    sig = np.zeros(n)
    for i in at:
        sig[i:i + 3] = [1.0, 3.0, 1.0]
    return sig + np.random.default_rng(0).normal(0, 0.05, n)


def test_the_offset_is_recovered_from_a_shared_pattern_of_motion() -> None:
    a = bursts(200, [20, 55, 90, 140])
    b = np.roll(a, 7)  # camera B's clock reads 7 frames later for every event
    assert estimate_offset(a, b, fps=30) == 7 / 30
    assert estimate_offset(b, a, fps=30) == -7 / 30


def test_motion_signal_is_the_mean_frame_difference() -> None:
    frames = [np.zeros((4, 4), np.uint8), np.full((4, 4), 10, np.uint8), np.full((4, 4), 10, np.uint8)]
    assert list(motion_signal(frames)) == [10.0, 0.0]


def test_frames_are_paired_after_the_offset_and_a_dropped_one_has_no_partner() -> None:
    a = [0.0, 0.1, 0.2, 0.3]
    b = [0.5, 0.6, 0.8]  # reads 0.5 ahead; the frame for 0.2 was dropped
    pairs = pair_frames(a, b, offset_b=0.5, max_gap_s=0.05)
    assert pairs == [(0, 0), (1, 1), (2, None), (3, 2)]
