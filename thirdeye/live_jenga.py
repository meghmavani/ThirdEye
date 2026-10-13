"""The live Jenga session: two RGB-D cameras in, legality and alerts out.

    python -m thirdeye.live_jenga --spec specs/jenga.json \
        --camera-dir rec/cam0 --camera-dir rec/cam1 --save-rig rig.json --log game.json
    python -m thirdeye.live_jenga --spec specs/jenga.json \
        --camera-dir rec/cam0 --camera-dir rec/cam1 --rig rig.json --no-hands

Each camera is a directory of recorded `.npz` RGB-D frames (FR-CAL-9). The live iPhone
stream is `perception.depth.Record3DSource`, which yields the same frames; it is not
wired into this command yet because the access route (OQ-8) is still open.

Camera poses come from the table markers in one frame of each camera (`--pose-frame`),
or from a rig saved by an earlier run (`--rig`, `--save-rig`) so a mounted rig need not
be recalibrated every session (HW-C3). The cameras are walked in lockstep, paired by
timestamp after the clock offset (`--offset`, seconds camera 1 reads ahead of camera 0;
`sync.estimate_offset` measures it); a frame with no partner inside one frame interval
runs single-camera (FR-CAL-10).

On exit the session is finished, the log written, and the per-stage latency table
printed so the NFR-PERF budget is measured on the machine that will run the demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from thirdeye.compiler.schema import ConstraintSpec, load_spec
from thirdeye.perception.camera_model import PinholeCamera, TableMarkers, pose_from_markers
from thirdeye.perception.depth import RecordedRGBD, RGBDFrame
from thirdeye.perception.jenga_pipeline import HandCounter, JengaPipeline
from thirdeye.perception.sync import pair_frames
from thirdeye.perception.tower_view import TowerReader, tower_region_px
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord


def make_hand_counter(cameras: list[PinholeCamera]) -> HandCounter:
    """MediaPipe hands counted inside each camera's tower outline; the most over the
    cameras is the count (a hand hidden from one view may show in the other)."""
    from thirdeye.perception.hands import HandDetector, hands_in_contact

    detector = HandDetector()
    regions = [tower_region_px(c) for c in cameras]

    def count(frames: list[RGBDFrame | None]) -> int | None:
        seen = [
            hands_in_contact(detector.detect(f.color, f.t), region)
            for f, region in zip(frames, regions, strict=True)
            if f is not None
        ]
        return max(seen) if seen else None

    return count


def calibrate(frames: list[RGBDFrame], markers: TableMarkers) -> list[PinholeCamera]:
    return [pose_from_markers(f.color, f.k, markers) for f in frames]


def save_rig(path: Path, cameras: list[PinholeCamera]) -> None:
    path.write_text(json.dumps([c.to_json() for c in cameras], indent=1) + "\n", encoding="utf-8")


def load_rig(path: Path) -> list[PinholeCamera]:
    return [PinholeCamera.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--camera-dir", type=Path, action="append", required=True,
                    help="a directory of recorded RGB-D frames; give one per camera")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds camera 1's clock reads ahead of camera 0's")
    ap.add_argument("--pose-frame", type=int, default=0, help="frame index to calibrate on")
    ap.add_argument("--rig", type=Path, default=None, help="camera poses saved by --save-rig")
    ap.add_argument("--save-rig", type=Path, default=None, help="write the calibrated poses here")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--serve", type=int, default=None, metavar="PORT",
                    help="also serve the browser UI on this port (state and alerts only)")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, ConstraintSpec):
        ap.error("this loop runs the Jenga rulebook spec; see thirdeye.live_jigsaw for jigsaw")
    sources = [list(RecordedRGBD(d)) for d in args.camera_dir]
    if args.rig:
        cameras = load_rig(args.rig)
        if len(cameras) != len(sources):
            ap.error(f"{args.rig} has {len(cameras)} cameras but {len(sources)} were given")
    else:
        cameras = calibrate([s[args.pose_frame] for s in sources], TableMarkers.around_tower())
    if args.save_rig:
        save_rig(args.save_rig, cameras)

    session = Session(spec)
    pipe = JengaPipeline(
        session,
        TowerReader(cameras, layers=spec.tower.layers, slots=spec.tower.slots_per_layer),
        hand_counter=None if args.no_hands else make_hand_counter(cameras),
    )

    publisher = None
    if args.serve:
        from thirdeye.server.app import serve_in_thread

        publisher = serve_in_thread(args.serve)
        publisher.start(spec.procedure, "recorded cameras")
        print(f"UI on http://127.0.0.1:{args.serve}/", flush=True)

    lead = sources[0]
    dt = float(np.median(np.diff([f.t for f in lead]))) if len(lead) > 1 else 0.1
    partners = [
        pair_frames([f.t for f in lead], [f.t for f in other],
                    args.offset if i == 0 else 0.0, dt)
        for i, other in enumerate(sources[1:])
    ]
    t = 0.0
    for i, frame in enumerate(lead):
        frames: list[RGBDFrame | None] = [frame]
        for other, pairs in zip(sources[1:], partners, strict=True):
            j = pairs[i][1]
            frames.append(None if j is None else other[j])
        t = frame.t
        alerts = pipe.process(t, frames)
        for alert in alerts:
            print(alert, flush=True)
        if publisher is not None and pipe.readings:
            publisher.update(session, t)

    for alert in session.finish(t):
        print(alert)
    if publisher is not None:
        publisher.finish(session, t, "recorded cameras")
    print()
    print(session.summary())
    print(pipe.timings.table())
    if pipe.last is not None and pipe.last.unobserved:
        print(f"\n{len(pipe.last.unobserved)} slots were unobserved on the last reading "
              "(held, never judged)")
    if args.log:
        record = SessionRecord.of(session, meta={"spec": str(args.spec), "readings": pipe.readings})
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
