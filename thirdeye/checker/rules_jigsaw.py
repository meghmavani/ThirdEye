"""Matching and alert text for the piece grid (FR-CHK-3..6).

The jigsaw half of the checker, in the same taxonomy as the stud grid but on a
flat board where a placement is a piece, a cell and a quarter-turn:

    a pending step wants this cell, piece differs     -> wrong part     (FR-CHK-5)
    a pending step wants this cell, only rot differs  -> wrong rotation (FR-CHK-6)
    this piece is wanted, but in another cell         -> wrong position (FR-CHK-6)
    nothing wants it                                  -> extra part     (FR-CHK-6)
    a completed step's piece is gone                  -> removed part   (FR-CHK-6)

An attempt made wrongly sends the step to `error`, which counts as performed for
the steps depending on it (see `events.COMPLETED`), so one mistake is one alert:
the next step is not also called out of order and the session-end sweep does not
report the same step missing. Taking the wrong piece away withdraws the attempt.

Out-of-order and missed-step come from the engine's generic precondition loop and
session-end sweep, not from here: they are properties of the dependency graph.
Unidentified pieces never reach this file -- the tracker emits no event for them
(FR-JIG-4).
"""

from __future__ import annotations

from typing import NamedTuple

from thirdeye.compiler.schema import CellPose, LoweredAction, PieceRef, Precondition
from thirdeye.events import COMPLETED, OPEN, Alert, Event, Status

WRONG_PART = "wrong_part"
WRONG_POSITION = "wrong_position"
WRONG_ROTATION = "wrong_rotation"
EXTRA_PART = "extra_part"
REMOVED_PART = "removed_part"

_ATTEMPTS = frozenset({WRONG_PART, WRONG_POSITION, WRONG_ROTATION})

Cell = tuple[int, int]


def _cell(p: CellPose) -> Cell:
    return (p.col, p.row)


def _where(p: CellPose) -> str:
    return f"column {p.col + 1}, row {p.row + 1}"


class _Placement(NamedTuple):
    action: LoweredAction
    part: PieceRef
    pose: CellPose


class PieceGridPolicy:
    """The jigsaw half of the checker."""

    def __init__(self) -> None:
        #: Cells holding a wrong piece, and the step it was an attempt at.
        self._stand_ins: dict[Cell, str] = {}

    def match(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> LoweredAction | None:
        if not isinstance(ev.part, PieceRef) or not isinstance(ev.pose, CellPose):
            return None
        for cand in self._open(actions, status):
            if cand.part == ev.part and cand.pose == ev.pose:
                return cand.action
        return None

    def unmatched(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        alerts = self._classify(ev, actions, status)
        for alert in alerts:
            if alert.code in _ATTEMPTS and alert.target and isinstance(ev.pose, CellPose):
                status[alert.target] = "error"
                self._stand_ins[_cell(ev.pose)] = alert.target
        return alerts

    def _classify(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        part, pose = ev.part, ev.pose
        if not isinstance(part, PieceRef) or not isinstance(pose, CellPose):
            return []

        # Exactly the piece of a step already credited: a re-read confirmed it
        # before the tracker committed it. The same piece seen twice is not extra.
        for action in actions:
            if status.get(action.id) in COMPLETED and action.part == part and action.pose == pose:
                return []

        open_actions = self._open(actions, status)

        at_cell = [c for c in open_actions if _cell(c.pose) == _cell(pose)]
        if at_cell:
            same_part = next((c for c in at_cell if c.part == part), None)
            if same_part is not None:
                return [
                    Alert(
                        t=ev.t,
                        code=WRONG_ROTATION,
                        rule="jigsaw.wrong_rotation",
                        target=same_part.action.id,
                        text=(
                            f"Piece {part} is in the right place ({_where(pose)}) but turned "
                            f"{pose.rot} degrees; step {same_part.action.id} needs it at "
                            f"{same_part.pose.rot}: \"{same_part.action.instruction}\""
                        ),
                    )
                ]
            wanted = at_cell[0]
            return [
                Alert(
                    t=ev.t,
                    code=WRONG_PART,
                    rule="jigsaw.wrong_part",
                    target=wanted.action.id,
                    text=(
                        f"That is piece {part} at {_where(pose)}, but step "
                        f"{wanted.action.id} needs piece {wanted.part} there: "
                        f"\"{wanted.action.instruction}\""
                    ),
                )
            ]

        wants_part = [c for c in open_actions if c.part == part]
        if wants_part:
            wanted = wants_part[0]
            return [
                Alert(
                    t=ev.t,
                    code=WRONG_POSITION,
                    rule="jigsaw.wrong_position",
                    target=wanted.action.id,
                    text=(
                        f"Piece {part} is at {_where(pose)}, but step {wanted.action.id} "
                        f"needs it at {_where(wanted.pose)}: \"{wanted.action.instruction}\""
                    ),
                )
            ]

        return [
            Alert(
                t=ev.t,
                code=EXTRA_PART,
                rule="jigsaw.extra_part",
                text=f"Piece {part} is on the board at {_where(pose)}, but no remaining step uses it.",
            )
        ]

    def on_remove(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """A piece leaving a completed step's cell undoes that step."""
        if not isinstance(ev.pose, CellPose):
            return []
        attempted = self._stand_ins.pop(_cell(ev.pose), None)
        if attempted is not None:
            if status.get(attempted) == "error":
                status[attempted] = "pending"
            return []
        for action in actions:
            if not isinstance(action.pose, CellPose) or _cell(action.pose) != _cell(ev.pose):
                continue
            if status.get(action.id) not in COMPLETED:
                continue
            status[action.id] = "pending"
            return [
                Alert(
                    t=ev.t,
                    code=REMOVED_PART,
                    rule="jigsaw.removed_part",
                    target=action.id,
                    text=(
                        f"Piece {action.part} from step {action.id} has been taken off "
                        f"{_where(action.pose)}; that step is no longer done."
                    ),
                )
            ]
        return []

    def matches_hint(self, ev: Event, action: LoweredAction) -> bool:
        return isinstance(ev.part, PieceRef) and action.part == ev.part

    def explain(self, pre: Precondition, action: LoweredAction, ev: Event) -> str:
        return f"{pre.text.capitalize()}. Step {action.id} is \"{action.instruction}\"."

    def missed_text(self, action: LoweredAction | None, action_id: str) -> str:
        if action is None or not isinstance(action.pose, CellPose):
            return f"Step {action_id} was never completed."
        return (
            f"Step {action_id} was never done: piece {action.part} is missing from "
            f"{_where(action.pose)} -- \"{action.instruction}\""
        )

    def _open(self, actions: list[LoweredAction], status: dict[str, Status]) -> list[_Placement]:
        attempted = set(self._stand_ins.values())
        return [
            _Placement(a, a.part, a.pose)
            for a in actions
            if isinstance(a.part, PieceRef)
            and isinstance(a.pose, CellPose)
            and (
                status.get(a.id) in OPEN
                or (status.get(a.id) == "error" and a.id in attempted)
            )
        ]
