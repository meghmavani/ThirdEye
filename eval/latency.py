"""Per-stage latency against the section 3.1 budget (EV-LAT-1, NFR-PERF-*).

    python -m eval.latency [--repeat 30] [--json out.json]

Times each stage this machine can exercise end to end, on synthetic input of the real
shape (a 256 x 192 depth map per camera, a 4 x 3 jigsaw board), and prints it beside
its budget. Run it on the demo laptop: the numbers here describe the machine that ran
them and nothing else. Stages that need hardware -- Record3D decode (NFR-PERF-12),
MediaPipe hands (NFR-PERF-4), the WebSocket push (NFR-PERF-10) -- are listed as
*not measured* rather than omitted, so the table cannot be read as complete when it
is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from eval.synth_jigsaw import box_picture, camera_view, flat_board, layout
from eval.synth_tower import corner_camera, full, render
from thirdeye.compiler.jigsaw import piece_id
from thirdeye.compiler.schema import load_spec
from thirdeye.events import Event, Slot
from thirdeye.perception.board import BoardView, rectify
from thirdeye.perception.calibrate import Calibrator
from thirdeye.perception.piece_match import PieceMatcher
from thirdeye.perception.tower_view import SlotReader
from thirdeye.session import Session

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Row:
    stage: str
    req: str
    budget_ms: float | None
    median_ms: float | None
    p95_ms: float | None

    @property
    def verdict(self) -> str:
        if self.median_ms is None:
            return "not measured"
        if self.budget_ms is None:
            return "-"
        return "ok" if (self.p95_ms or self.median_ms) <= self.budget_ms else "OVER"


def _time(fn: Callable[[], object], repeat: int) -> tuple[float, float]:
    fn()  # warm-up: first calls pay imports and cache fills
    ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return float(median(ms)), float(np.percentile(ms, 95))


def measure(repeat: int = 30) -> list[Row]:
    rows: list[Row] = []

    def add(stage: str, req: str, budget: float | None, fn: Callable[[], object], scale: float = 1.0):
        m, p = _time(fn, repeat)
        rows.append(Row(stage, req, budget, m * scale, p * scale))

    # Jenga slot reading, both cameras, every frame.
    cams = [corner_camera(-1, -1), corner_camera(1, 1)]
    frames = [render(c, full()) for c in cams]
    reader = SlotReader(cams, layers=22)
    add("Jenga slot reading, 2 cameras", "NFR-PERF-3", 15.0, lambda: reader.read(frames))

    # Jigsaw: board read + match on hand-exit.
    cols, rows_ = 4, 3
    box = box_picture(cols, rows_)
    cal = Calibrator(__import__("tests.jigsaw_layout", fromlist=["x"]).layout(cols, rows_)) \
        if False else None
    from eval.synth_jigsaw import layout

    cal = Calibrator(layout(cols, rows_))
    empty = camera_view(flat_board(box, cols, rows_))
    h = cal.update(empty).homography
    view = BoardView(PieceMatcher(box, cols, rows_))
    view.learn_empty(rectify(empty, h, cols, rows_, view.cell_px))
    done = camera_view(flat_board(box, cols, rows_, [(piece_id(c, r), c, r, 0)
                                                     for r in range(rows_) for c in range(cols)]))
    add("Jigsaw rectify + match, 12 pieces", "NFR-PERF-13", 100.0,
        lambda: view.read(rectify(done, h, cols, rows_, view.cell_px)))

    # Homography refresh, amortised over the refresh interval.
    add("Homography refresh (amortised /10)", "NFR-PERF-6", 1.0,
        lambda: cal.refresh(empty), scale=1 / 10)

    # State diff + checker per event.
    spec = load_spec(ROOT / "specs" / "jenga.json")
    session = Session(spec)
    state = session.state
    ev = Event(t=1.0, kind="MOVE", src=Slot(5, 1), dst=Slot(18, 0), hands_in_contact=1)
    add("Checker, per event (Jenga move)", "NFR-PERF-8", 1.0, lambda: session.checker.on_event(ev))
    add("State diff, no change", "NFR-PERF-7", 1.0, state.diff)

    for stage, req in (
        ("Frame + depth decode (Record3D)", "NFR-PERF-12"),
        ("MediaPipe hands", "NFR-PERF-4"),
        ("WebSocket overlay push", "NFR-PERF-10"),
        ("Temporal model, async", "NFR-PERF-9"),
    ):
        rows.append(Row(stage, req, None, None, None))
    return rows


def table(rows: list[Row]) -> str:
    out = ["| stage | requirement | budget ms | median ms | p95 ms | |", "| --- | --- | --: | --: | --: | --- |"]
    for r in rows:
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        out.append(f"| {r.stage} | {r.req} | {f(r.budget_ms)} | {f(r.median_ms)} | {f(r.p95_ms)} | {r.verdict} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    rows = measure(args.repeat)
    print(table(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([vars(r) | {"verdict": r.verdict} for r in rows], f, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
