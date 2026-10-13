"""The live session: camera in, checklist and alerts out (FR-SYS-1, FR-UI-4, plan 4.5).

    python -m thirdeye.live --spec specs/model_a.json --zero-shot
    python -m thirdeye.live --spec specs/model_a.json --weights runs/detect/best.pt
    python -m thirdeye.live --spec specs/model_a.json --zero-shot --video run07.mp4 --no-show

The camera is read here, in the Python process (FR-UI-4); nothing travels to a
browser but what is drawn. This is the terminal-and-window version of the demo:
the same session the WebSocket server will drive, with an OpenCV window for the
overlay -- plate outline, detections, hands, a ghost of the next step, the
checklist and the alerts.

On exit (q, Ctrl-C, end of video) the session is finished, the missed-step
sweep runs, the log is written (IF-LOG-1), and the per-stage latency table is
printed so the plan 3.1 budget is measured on the machine that will run the
demo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.schema import DagSpec, load_spec
from thirdeye.perception.calibrate import Calibrator, MarkerLayout
from thirdeye.perception.detect import Detector, YoloDetector, ZeroShotDetector
from thirdeye.perception.hands import DEFAULT_MODEL, HandDetector

MODELS = DEFAULT_MODEL.parent
from thirdeye.perception.pipeline import FrameResult, TopDownPipeline
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import StudGrid, extent

RED, YELLOW, GREEN, CYAN, WHITE = (40, 40, 230), (0, 210, 255), (60, 200, 60), (230, 200, 0), \
    (240, 240, 240)


def _to_px(inv: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, inv).reshape(-1, 2).astype(np.int32)


def draw(frame: np.ndarray, result: FrameResult, session: Session, recent: list[str],
         grid: StudGrid) -> np.ndarray:
    """The overlay (FR-UI-1..3, FR-UI-5), drawn onto a copy of the frame."""
    out = frame.copy()
    cal = result.calibration
    status_colour = GREEN if cal.status == "ok" else YELLOW if cal.usable else RED
    cv2.putText(out, f"calibration: {cal.status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_colour, 2)
    if result.hands_present:
        cv2.putText(out, "hand over plate - frozen", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    YELLOW, 2)

    if cal.usable and cal.homography is not None:
        inv = np.linalg.inv(cal.homography)
        plate = _to_px(inv, [(0, 0), (grid.studs_x, 0), (grid.studs_x, grid.studs_y),
                             (0, grid.studs_y)])
        cv2.polylines(out, [plate], True, CYAN, 1)
        nxt = session.checker.next_expected()
        if nxt is not None and nxt.part is not None and nxt.pose is not None:
            w, h = extent(nxt.part, nxt.pose.rot)
            p = nxt.pose
            ghost = _to_px(inv, [(p.x, p.y), (p.x + w, p.y), (p.x + w, p.y + h), (p.x, p.y + h)])
            cv2.polylines(out, [ghost], True, WHITE, 2, cv2.LINE_AA)
            cv2.putText(out, f"next: {nxt.id} {nxt.part}", tuple(int(v) for v in ghost[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    for det in result.detections:
        x0, y0, x1, y1 = (int(v) for v in det.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), GREEN, 1)
    for hand in result.hands:
        for x, y in hand.keypoints.astype(int):
            cv2.circle(out, (int(x), int(y)), 2, YELLOW, -1)

    # Checklist and the latest alerts down the right-hand side.
    panel_x = out.shape[1] - 330
    cv2.rectangle(out, (panel_x - 10, 0), (out.shape[1], out.shape[0]), (20, 20, 20), -1)
    y = 22
    marks = {"done": ("[x]", GREEN), "error": ("[!]", RED), "active": ("[~]", YELLOW),
             "pending": ("[ ]", WHITE)}
    for step_id, status, _instruction in session.checklist():
        mark, colour = marks[status]
        cv2.putText(out, f"{mark} {step_id}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1)
        y += 18
    y += 10
    for line in recent[-6:]:
        colour = RED if " ! " in line else YELLOW
        cv2.putText(out, line[:48], (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        y += 16
    return out


def build_detector(args: argparse.Namespace) -> Detector:
    if args.weights:
        return YoloDetector(args.weights, conf=args.conf, device=args.device)
    return ZeroShotDetector(args.zero_shot_weights, conf=args.conf, device=args.device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", type=Path, help="a recorded run instead of a live camera")
    det = ap.add_mutually_exclusive_group(required=True)
    det.add_argument("--weights", help="fine-tuned YOLO weights (class names = part names)")
    det.add_argument("--zero-shot", action="store_true", help="YOLO-World fallback (FR-PER-3)")
    ap.add_argument("--zero-shot-weights", default=str(MODELS / "yolov8s-worldv2.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu; default picks the GPU")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--log", type=Path, default=None, help="session log path (IF-LOG-1)")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, DagSpec):
        ap.error("the live top-down loop runs placement procedures; Jenga needs the side rig")
    grid = StudGrid.from_spec(spec.baseplate)
    session = Session(spec)
    assert isinstance(session.state, BuildState)
    pipeline = TopDownPipeline(
        Calibrator(MarkerLayout.around_plate(grid.studs_x, grid.studs_y)),
        build_detector(args),
        grid,
        hands=None if args.no_hands else HandDetector(),
    )

    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        print(f"cannot open {'video ' + str(args.video) if args.video else 'camera'}",
              file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, frames, recent = time.monotonic(), 0, []
    t, blind = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # A file is timed by its frames, a camera by the clock.
            t = frames / fps if args.video else time.monotonic() - start
            frames += 1
            result = pipeline.process(frame, t)
            blind += not result.usable
            if result.usable:
                session.state.observe(t, result.readings, hands_present=result.hands_present)
                for alert in session.pump():
                    print(alert, flush=True)
                    recent.append(str(alert))
            if not args.no_show:
                cv2.imshow("ThirdEye", draw(frame, result, session, recent, grid))
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
    if blind:
        print(f"\n{blind} of {frames} frames had no usable calibration and were not judged"
              " -- are all four ArUco markers in view? (FR-CAL-7)")
    if args.log:
        record = SessionRecord.of(session, meta={"spec": str(args.spec), "frames": frames})
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
