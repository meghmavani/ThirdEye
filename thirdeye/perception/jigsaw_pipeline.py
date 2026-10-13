"""One overhead jigsaw frame in, piece readings out (plan 4.5, steps 2-4).

    frame -> calibration -> hands -> rectify -> occupied cells -> piece matching

Matching is the expensive stage and the one that only matters once the tracker
will use the answer, so it runs only on frames with no hand over the board
(FR-PER-6, NFR-PERF-13). Calibration comes first for the same reason as in the
top-down LEGO pipeline: without a usable homography there are no cell coordinates
to give, and a reading taken through a stale fit is worse than none (FR-CAL-7).

The empty board is learned from the first usable, hand-free frame
(`auto_empty`), so the operator starts a session with the board clear. The
puzzle's pieces are placed on a plain mat (HW-6); segmentation is a difference
against that picture (see `board.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from thirdeye.perception.board import BoardView, rectify
from thirdeye.perception.calibrate import Calibration, Calibrator
from thirdeye.perception.hands import Hand, HandDetector, over_workspace
from thirdeye.perception.pipeline import Timings
from thirdeye.state.jigsaw.piece_grid import PieceReading


@dataclass
class JigsawFrame:
    t: float
    calibration: Calibration
    hands: list[Hand] = field(default_factory=list)
    hands_present: bool = False
    readings: list[PieceReading] = field(default_factory=list)
    #: True on the frame the empty board was learned; the tracker should see
    #: nothing from it.
    learned_empty: bool = False

    @property
    def usable(self) -> bool:
        return self.calibration.usable and not self.learned_empty


@dataclass
class JigsawPipeline:
    calibrator: Calibrator
    view: BoardView
    hands: HandDetector | None = None
    auto_empty: bool = True
    timings: Timings = field(default_factory=Timings)

    def process(self, frame_bgr: np.ndarray, t: float) -> JigsawFrame:
        clock = time.perf_counter
        start = clock()
        cal = self.calibrator.update(frame_bgr)
        out = JigsawFrame(t=t, calibration=cal)
        if not cal.usable or cal.homography is None:
            self.timings.add("total", clock() - start)
            return out
        h = cal.homography.tolist()

        if self.hands is not None:
            t0 = clock()
            out.hands = self.hands.detect(frame_bgr, t)
            out.hands_present = any(
                over_workspace(hand, h, self.view.cols, self.view.rows) for hand in out.hands
            )
            self.timings.add("hands", clock() - t0)

        if out.hands_present:
            self.timings.add("total", clock() - start)
            return out

        t0 = clock()
        board = rectify(frame_bgr, cal.homography, self.view.cols, self.view.rows,
                        self.view.cell_px)
        self.timings.add("rectify", clock() - t0)
        if self.view.empty is None:
            if self.auto_empty:
                self.view.learn_empty(board)
                out.learned_empty = True
            self.timings.add("total", clock() - start)
            return out

        t0 = clock()
        out.readings = self.view.read(board)
        self.timings.add("match", clock() - t0)
        self.timings.add("total", clock() - start)
        return out
