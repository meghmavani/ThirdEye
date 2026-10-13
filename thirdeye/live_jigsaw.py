"""The live jigsaw session: overhead webcam in, checklist and alerts out (FR-SYS-1, FR-UI-4).

    python -m thirdeye.live_jigsaw --spec specs/jigsaw_forest_12.json --box forest_box.png
    python -m thirdeye.live_jigsaw --spec ... --box ... --video run03.mp4 --no-show --log run.json

Start with the board clear: the first frame with all four markers in view and no
hand over the board is taken as the empty board. The camera is read in this
process (FR-UI-4). On exit the session is finished, the missed-step sweep runs,
the log is written and the per-stage latency table is printed (NFR-PERF-13).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.schema import JigsawSpec, load_spec
from thirdeye.perception.board import BoardView
from thirdeye.perception.calibrate import Calibrator, MarkerLayout
from thirdeye.perception.hands import HandDetector
from thirdeye.perception.jigsaw_pipeline import JigsawFrame, JigsawPipeline
from thirdeye.perception.piece_match import PieceMatcher
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.jigsaw.piece_grid import PieceGridState

RED, YELLOW, GREEN, CYAN, WHITE = (40, 40, 230), (0, 210, 255), (60, 200, 60), (230, 200, 0), \
    (240, 240, 240)

#: Printed markers are one cell wide with a small gap (see `scripts/make_markers.py`).
MARKER_SIDE_CELLS = 1.0
MARKER_GAP_CELLS = 0.4


def _to_px(inv: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, inv).reshape(-1, 2).astype(np.int32)


def draw(frame: np.ndarray, result: JigsawFrame, session: Session, recent: list[str],
         spec: JigsawSpec) -> np.ndarray:
    """The overlay (FR-UI-1..3, FR-UI-5): grid, identified pieces, ghost, checklist, alerts."""
    out = frame.copy()
    cal = result.calibration
    colour = GREEN if cal.status == "ok" else YELLOW if cal.usable else RED
    cv2.putText(out, f"calibration: {cal.status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                colour, 2)
    if result.hands_present:
        cv2.putText(out, "hand over board - frozen", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    YELLOW, 2)
    cols, rows = spec.grid.cols, spec.grid.rows
    if cal.usable and cal.homography is not None:
        inv = np.linalg.inv(cal.homography)
        for c in range(cols + 1):
            cv2.polylines(out, [_to_px(inv, [(c, 0), (c, rows)])], False, CYAN, 1)
        for r in range(rows + 1):
            cv2.polylines(out, [_to_px(inv, [(0, r), (cols, r)])], False, CYAN, 1)
        for reading in result.readings:
            quad = _to_px(inv, [(reading.col, reading.row), (reading.col + 1, reading.row),
                                (reading.col + 1, reading.row + 1), (reading.col, reading.row + 1)])
            cv2.polylines(out, [quad], True, GREEN if reading.piece else YELLOW, 2)
            label = (reading.piece or "?") + (f" {reading.rot}" if reading.rot else "")
            cv2.putText(out, label, tuple(int(v) for v in quad[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        nxt = session.checker.next_expected()
        if nxt is not None and nxt.pose is not None and hasattr(nxt.pose, "col"):
            p = nxt.pose
            ghost = _to_px(inv, [(p.col, p.row), (p.col + 1, p.row), (p.col + 1, p.row + 1),
                                 (p.col, p.row + 1)])
            cv2.polylines(out, [ghost], True, WHITE, 2, cv2.LINE_AA)
            cv2.putText(out, f"next: {nxt.id} {nxt.part}", tuple(int(v) for v in ghost[2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    panel_x = out.shape[1] - 330
    cv2.rectangle(out, (panel_x - 10, 0), (out.shape[1], out.shape[0]), (20, 20, 20), -1)
    y = 22
    marks = {"done": ("[x]", GREEN), "error": ("[!]", RED), "active": ("[~]", YELLOW),
             "pending": ("[ ]", WHITE)}
    for step_id, status, _ in session.checklist():
        mark, col = marks[status]
        cv2.putText(out, f"{mark} {step_id}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y += 18
    y += 10
    for line in recent[-6:]:
        cv2.putText(out, line[:48], (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    RED if " ! " in line else YELLOW, 1)
        y += 16
    return out


def build_pipeline(spec: JigsawSpec, box: np.ndarray, hands: bool, cell_px: int = 64) -> JigsawPipeline:
    layout = MarkerLayout.around_plate(
        spec.grid.cols, spec.grid.rows, side_studs=MARKER_SIDE_CELLS, gap_studs=MARKER_GAP_CELLS
    )
    return JigsawPipeline(
        Calibrator(layout),
        BoardView(PieceMatcher(box, spec.grid.cols, spec.grid.rows), cell_px=cell_px),
        hands=HandDetector() if hands else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--box", type=Path, required=True, help="the box picture, cropped to the puzzle")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", type=Path)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, JigsawSpec):
        ap.error("this loop runs jigsaw specs; see thirdeye.live for LEGO and thirdeye.live_jenga")
    box = cv2.imread(str(args.box))
    if box is None:
        ap.error(f"cannot read the box picture {args.box}")

    session = Session(spec)
    assert isinstance(session.state, PieceGridState)
    pipeline = build_pipeline(spec, box, hands=not args.no_hands)

    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        print("cannot open the video source", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, frames, recent, t, blind = time.monotonic(), 0, [], 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frames / fps if args.video else time.monotonic() - start
            frames += 1
            result = pipeline.process(frame, t)
            blind += not result.calibration.usable
            if result.usable:
                session.state.observe(t, result.readings, hands_present=result.hands_present)
                for alert in session.pump():
                    print(alert, flush=True)
                    recent.append(str(alert))
            if not args.no_show:
                cv2.imshow("ThirdEye", draw(frame, result, session, recent, spec))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds is not None and t >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    for alert in session.finish(t):
        print(alert)
    elapsed = time.monotonic() - start
    print()
    print(session.summary())
    print(f"\n{frames} frames in {elapsed:.1f} s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    print(pipeline.timings.table())
    unid = session.state.unidentified_seen
    if unid:
        print(f"\n{unid} cell-readings were unidentified (never alerted on)")
    if blind:
        print(f"\n{blind} of {frames} frames had no usable calibration -- are all four "
              "markers in view? (FR-CAL-7)")
    if args.log:
        record = SessionRecord.of(session, meta={"spec": str(args.spec), "frames": frames})
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
