"""Replay a scripted session through the real tracker and checker.

    python -m thirdeye.replay scripts/scenarios/model_a_swap.yaml [--log out.json]

A script describes the *scene*, not the events: which bricks are on the plate,
whether a hand is over it, which Jenga block was pulled. The replayer films
that scene at a fixed frame rate and feeds each frame to the same `observe` a
camera would, so the stability filter, the occlusion freeze and the move pairer
all do their real work. Scripting events directly would test the checker and
nothing else; the checker already has unit tests.

This is what makes the planted-error protocol of plan 5.1 runnable before any
hardware exists, and it is the harness the live path is checked against later:
a recorded run, once labelled, is just a script with real timings.

Script format (YAML)::

    spec: ../../specs/example_dag.json   # relative to the script
    fps: 10                              # optional, default 10
    end: 12.0                            # optional, default last keyframe + 2 s
    keyframes:
      - t: 0.0
        bricks: []                       # the whole LEGO scene, persists
      - t: 1.0
        hands: true                      # persists until changed
      - t: 2.0
        hands: false
        bricks: ["red_2x4 @ 0,0", "blue_2x2 @ 0,2,1 conf=0.3"]
      - t: 3.0
        add: ["yellow_1x4 @ 4,4,0,90"]   # or edit the running scene
        remove: ["red_2x4 @ 0,0"]         # by cell
      - t: 3.0                           # Jenga: edits to the running tower
        pull: [5, 1]
        hands: 1                         # most hands on the tower since last reading
      - t: 6.0
        place: [18, 0]
      - t: 7.0
        collapse: true
      - t: 4.0                           # a one-shot event, e.g. a temporal hint
        event: {kind: STEP_START, source: temporal, part: red_2x4}

A brick is `color_WxH @ x,y[,layer[,rot]]` with an optional `conf=`.

A jigsaw scene (`pieces:` / `add:` / `remove:`, same shape as bricks) is made of
`r<row>c<col> @ col,row[,rot]` entries, also with an optional `conf=`; `? @ col,row`
is a piece on the board the matcher could not identify.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from thirdeye.compiler.schema import CellPose, PartRef, PieceRef, Pose, load_spec
from thirdeye.events import Alert, Event, Slot
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.jenga.lattice import TowerLattice, full_tower
from thirdeye.state.jigsaw.piece_grid import PieceGridState, PieceReading
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import Reading

_BRICK = re.compile(
    r"^\s*(?P<color>[a-z]+)_(?P<size>\d+x\d+)\s*@\s*(?P<pose>[\d,\s]+?)"
    r"(?:\s+conf=(?P<conf>[\d.]+))?\s*$"
)


_PIECE = re.compile(
    r"^\s*(?P<piece>\?|r\d+c\d+)\s*@\s*(?P<pose>[\d,\s]+?)"
    r"(?:\s+conf=(?P<conf>[\d.]+))?\s*$"
)


class ScriptError(ValueError):
    """The script itself is malformed -- the author's bug, not a finding."""


def part(text: str) -> PartRef:
    """`red_2x4` -> PartRef, the detector's own class naming (FR-PER-2)."""
    color, _, size = text.partition("_")
    return PartRef(color=color, size=size)


def brick(text: str) -> Reading:
    """Parse one brick of a LEGO scene into the reading a detector would give."""
    m = _BRICK.match(text)
    if m is None:
        raise ScriptError(f"cannot read brick {text!r}; expected 'red_2x4 @ x,y[,layer[,rot]]'")
    nums = [int(n) for n in m["pose"].replace(" ", "").split(",") if n]
    if not 2 <= len(nums) <= 4:
        raise ScriptError(f"brick {text!r} needs x,y and optionally layer and rot")
    x, y, layer, rot = (*nums, 0, 0)[:4]
    return Reading(
        part=PartRef(color=m["color"], size=m["size"]),
        pose=Pose(x=x, y=y, layer=layer, rot=rot),  # type: ignore[arg-type]
        residual=0.0,
        confidence=float(m["conf"]) if m["conf"] else 1.0,
    )


def piece(text: str) -> PieceReading:
    """Parse one piece of a jigsaw scene into the reading the board view would give."""
    m = _PIECE.match(text)
    if m is None:
        raise ScriptError(f"cannot read piece {text!r}; expected 'r0c1 @ col,row[,rot]'")
    nums = [int(n) for n in m["pose"].replace(" ", "").split(",") if n]
    if not 2 <= len(nums) <= 3:
        raise ScriptError(f"piece {text!r} needs col,row and optionally rot")
    col, row, rot = (*nums, 0)[:3]
    return PieceReading(
        col=col,
        row=row,
        piece=None if m["piece"] == "?" else m["piece"],
        rot=rot,
        confidence=float(m["conf"]) if m["conf"] else 1.0,
    )


def _slots(value: Any) -> list[Slot]:
    """`[5, 1]` or `[[5, 1], [6, 0]]` -> slots."""
    if not value:
        return []
    if isinstance(value[0], int):
        value = [value]
    return [Slot(int(layer), int(slot)) for layer, slot in value]


def _event(data: dict[str, Any], t: float) -> Event:
    fields = dict(data)
    if "part" in fields:
        fields["part"] = (
            PieceRef(piece=fields["part"])
            if re.fullmatch(r"r\d+c\d+", fields["part"])
            else part(fields["part"])
        )
    if "pose" in fields:
        fields["pose"] = (
            CellPose(**fields["pose"]) if "col" in fields["pose"] else Pose(**fields["pose"])
        )
    for key in ("src", "dst"):
        if key in fields:
            fields[key] = Slot(*fields[key])
    return Event(t=float(fields.pop("t", t)), **fields)


@dataclass
class Replay:
    """A session driven from a script instead of a camera."""

    session: Session
    keyframes: list[dict[str, Any]]
    fps: float = 10.0
    end: float | None = None
    #: Alerts in the order they were raised, for the demo printout.
    raised: list[Alert] = field(default_factory=list, init=False)

    @classmethod
    def load(cls, path: str | Path) -> Replay:
        path = Path(path)
        script = yaml.safe_load(path.read_text())
        if not isinstance(script, dict) or "spec" not in script:
            raise ScriptError(f"{path}: a script needs a 'spec' and 'keyframes'")
        spec = load_spec((path.parent / script["spec"]).resolve())
        frames = sorted(script.get("keyframes", []), key=lambda k: float(k["t"]))
        return cls(
            session=Session(spec),
            keyframes=frames,
            fps=float(script.get("fps", 10.0)),
            end=script.get("end"),
        )

    def run(self) -> SessionRecord:
        if isinstance(self.session.state, BuildState):
            self._film_lego()
        elif isinstance(self.session.state, PieceGridState):
            self._film_jigsaw()
        elif isinstance(self.session.state, TowerLattice):
            self._film_jenga()
        else:  # pragma: no cover - a new state model needs a filming rule here
            raise ScriptError(f"cannot replay onto {type(self.session.state).__name__}")
        end = self._end()
        self.raised += self.session.finish(end)
        return SessionRecord.of(self.session)

    def _end(self) -> float:
        if self.end is not None:
            return float(self.end)
        return (float(self.keyframes[-1]["t"]) if self.keyframes else 0.0) + 2.0

    def _film_lego(self) -> None:
        """Film the scene at `fps`, holding each keyframe until the next."""
        state = self.session.state
        assert isinstance(state, BuildState)
        bricks: list[Reading] = []
        hands = False
        dt = 1.0 / self.fps
        stops = [float(k["t"]) for k in self.keyframes[1:]] + [self._end()]
        for key, stop in zip(self.keyframes, stops, strict=True):
            t = float(key["t"])
            if "bricks" in key:
                bricks = [brick(b) for b in key["bricks"] or []]
            gone = {brick(b).cell for b in key.get("remove") or []}
            bricks = [b for b in bricks if b.cell not in gone]
            bricks += [brick(b) for b in key.get("add") or []]
            hands = bool(key.get("hands", hands))
            if "event" in key:
                self.raised += self.session.submit(_event(key["event"], t))
            while t < stop - 1e-9:
                state.observe(t, bricks, hands_present=hands)
                self.raised += self.session.pump()
                t += dt

    def _film_jigsaw(self) -> None:
        """Film a jigsaw scene exactly as `_film_lego` films bricks."""
        state = self.session.state
        assert isinstance(state, PieceGridState)
        pieces: list[PieceReading] = []
        hands = False
        dt = 1.0 / self.fps
        stops = [float(k["t"]) for k in self.keyframes[1:]] + [self._end()]
        for key, stop in zip(self.keyframes, stops, strict=True):
            t = float(key["t"])
            if "pieces" in key:
                pieces = [piece(p) for p in key["pieces"] or []]
            gone = {piece(p).cell for p in key.get("remove") or []}
            pieces = [p for p in pieces if p.cell not in gone]
            pieces += [piece(p) for p in key.get("add") or []]
            hands = bool(key.get("hands", hands))
            if "event" in key:
                self.raised += self.session.submit(_event(key["event"], t))
            while t < stop - 1e-9:
                state.observe(t, pieces, hands_present=hands)
                self.raised += self.session.pump()
                t += dt

    def _film_jenga(self) -> None:
        """A lattice reading per keyframe that changes the tower.

        The lattice has no stability filter of its own -- perception hands it a
        settled reading after the hands leave -- so there is nothing to film
        between keyframes.
        """
        state = self.session.state
        assert isinstance(state, TowerLattice)
        rows = [list(r) for r in full_tower(slots=state.slots_per_layer)]
        for key in self.keyframes:
            t = float(key["t"])
            if "event" in key:
                self.raised += self.session.submit(_event(key["event"], t))
            touched = False
            for s in _slots(key.get("pull")):
                rows[s.layer][s.slot] = 0
                touched = True
            for s in _slots(key.get("place")):
                while len(rows) <= s.layer:
                    rows.append([0] * state.slots_per_layer)
                rows[s.layer][s.slot] = 1
                touched = True
            if key.get("collapse"):
                state.collapse_detected = True
                touched = True
            if touched:
                hands = key.get("hands")
                state.observe(rows, t=t, hands_in_contact=None if hands is None else int(hands))
                self.raised += self.session.pump()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("script", type=Path)
    ap.add_argument("--log", type=Path, help="write the session log (IF-LOG-1) here")
    ap.add_argument("-q", "--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)

    replay = Replay.load(args.script)
    record = replay.run()
    record.meta["script"] = str(args.script)
    if not args.quiet:
        for alert in replay.raised:
            print(alert)
        print()
    print(replay.session.summary())
    if args.log:
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

