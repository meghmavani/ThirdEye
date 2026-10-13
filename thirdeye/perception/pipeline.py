"""One top-down frame in, grid readings out (plan 4.5, steps 2-4).

    frame -> calibration -> hands -> parts -> readings on the stud grid

The order is deliberate. Calibration first, because without a usable
homography there are no stud coordinates to give and the frame is dropped
whole (FR-CAL-7) -- a reading taken through a stale fit is worse than none.
Hands before parts, because a hand over the plate means the tracker will not
use this frame's parts anyway (FR-PER-6), and the detector's 15-25 ms can be
skipped (NFR-PERF-2).

Each stage's time is recorded, so the latency budget of plan 3.1 is measured
on the demo machine rather than asserted.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from thirdeye.perception.calibrate import Calibration, Calibrator
from thirdeye.perception.detect import Detection, Detector, to_readings
from thirdeye.perception.hands import Hand, HandDetector, over_workspace
from thirdeye.state.lego.grid import Reading, StudGrid


@dataclass
class FrameResult:
    t: float
    calibration: Calibration
    hands: list[Hand] = field(default_factory=list)
    hands_present: bool = False
    detections: list[Detection] = field(default_factory=list)
    readings: list[Reading] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Should the tracker see this frame at all?"""
        return self.calibration.usable


@dataclass
class Timings:
    """Per-stage wall time, in milliseconds, for the plan 3.1 budget table."""

    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, stage: str, seconds: float) -> None:
        self.samples[stage].append(seconds * 1000)

    def table(self) -> str:
        rows = [f"{'stage':<12} {'median ms':>10} {'p95 ms':>8} {'n':>6}"]
        for stage, ms in self.samples.items():
            arr = np.asarray(ms)
            rows.append(f"{stage:<12} {np.median(arr):>10.1f} {np.percentile(arr, 95):>8.1f} "
                        f"{len(arr):>6}")
        return "\n".join(rows)


@dataclass
class TopDownPipeline:
    calibrator: Calibrator
    detector: Detector
    grid: StudGrid
    hands: HandDetector | None = None
    #: Run the detector even under a hand -- for the overlay and for recording,
    #: never for the tracker. Off on the demo path to save the frame budget.
    detect_under_hands: bool = False
    timings: Timings = field(default_factory=Timings)

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        clock = time.perf_counter
        start = clock()

        t0 = clock()
        cal = self.calibrator.update(frame_bgr)
        self.timings.add("calibrate", clock() - t0)
        result = FrameResult(t=t, calibration=cal)
        if not cal.usable or cal.homography is None:
            self.timings.add("total", clock() - start)
            return result
        h = cal.homography.tolist()

        if self.hands is not None:
            t0 = clock()
            result.hands = self.hands.detect(frame_bgr, t)
            result.hands_present = any(
                over_workspace(hand, h, self.grid.studs_x, self.grid.studs_y)
                for hand in result.hands
            )
            self.timings.add("hands", clock() - t0)

        if not result.hands_present or self.detect_under_hands:
            t0 = clock()
            result.detections = self.detector.detect(frame_bgr)
            self.timings.add("detect", clock() - t0)
            t0 = clock()
            result.readings = to_readings(result.detections, frame_bgr, h, self.grid)
            self.timings.add("to_grid", clock() - t0)

        self.timings.add("total", clock() - start)
        return result
