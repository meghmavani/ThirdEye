"""Time alignment between the Jenga cameras (FR-CAL-10, HW-C2).

The iPhone and the second camera stamp frames on different clocks, so a single pull
could be attributed to two moments a few frames apart and the move pairer would
see them as two events. Two small tools:

*`estimate_offset`* -- cross-correlate a motion signal (mean absolute frame
difference, one value per frame) from each camera and report the lag, in seconds,
that best aligns them. A clap, or the first pull of a recording, gives a sharp
peak. The sign convention: the result `d` means camera B's clock reads `d` seconds
*later* than camera A's for the same instant, so subtract it from B's stamps.

*`pair_frames`* -- match each A frame to the nearest B frame once B is shifted, and
refuse a pair further apart than one frame interval, so a dropped frame yields a
single-camera tick rather than a stale second view.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def motion_signal(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Mean absolute difference between consecutive frames, length n - 1."""
    out = []
    for a, b in zip(frames[:-1], frames[1:], strict=True):
        out.append(float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean()))
    return np.asarray(out)


def estimate_offset(
    signal_a: np.ndarray, signal_b: np.ndarray, fps: float, max_lag_s: float = 2.0
) -> float:
    """Seconds by which B's clock leads A's for the same event (see module docstring)."""
    a = np.asarray(signal_a, dtype=np.float64)
    b = np.asarray(signal_b, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    max_lag = int(max_lag_s * fps)
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        # lag > 0: the event appears `lag` samples later in B than in A
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        n = min(len(x), len(y))
        if n < 3:
            continue
        score = float(np.dot(x[:n], y[:n]) / n)
        if score > best:
            best, best_lag = score, lag
    return best_lag / fps


def pair_frames(
    times_a: Sequence[float],
    times_b: Sequence[float],
    offset_b: float,
    max_gap_s: float,
) -> list[tuple[int, int | None]]:
    """For each A frame, the index of the B frame nearest in corrected time, or None
    when none is within `max_gap_s` (a dropped frame)."""
    b = np.asarray(times_b, dtype=np.float64) - offset_b
    out: list[tuple[int, int | None]] = []
    for i, ta in enumerate(times_a):
        if len(b) == 0:
            out.append((i, None))
            continue
        j = int(np.argmin(np.abs(b - ta)))
        out.append((i, j if abs(b[j] - ta) <= max_gap_s else None))
    return out
