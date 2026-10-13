"""The jigsaw piece-grid tracker (FR-STA-3, FR-STA-4, FR-STA-6, FR-STA-10).

A `cols x rows` board of cells, each empty, holding an identified piece at a
rotation, or holding something we could not identify. Like the stud-grid tracker
it observes rather than simulates, and it has the same two guards against the
workspace's dominant failure mode, a hand over the board:

*Freeze while occluded* (FR-PER-6): nothing is committed while a hand is present
and the stability timers restart when it leaves.

*Commit only what has held still* (FR-STA-3): a piece must be seen continuously
for `stable_s` before it is placed, and missing continuously for `stable_s`
before it is removed -- per cell, so a flicker in one corner stalls nothing else.

**Unidentified pieces never produce events** (FR-JIG-4). A piece whose best match
is not clearly better than the second-best is *there* -- the cell is occupied --
but we do not know which piece it is, and a claim about the user cannot rest on
a guess. It is tracked so the cell is not read as empty, and an identified piece
that later turns unidentified is held in place rather than removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thirdeye.compiler.schema import CellPose, PieceRef
from thirdeye.events import Event
from thirdeye.state.base import WorldState

Cell = tuple[int, int]  # (col, row)


@dataclass(frozen=True)
class PieceReading:
    """One cell's reading from the board view.

    `piece` is None for an unidentified piece. `margin` is the matcher's gap
    between its best and second-best candidate, carried for the perception report.
    """

    col: int
    row: int
    piece: str | None
    rot: int = 0
    confidence: float = 1.0
    margin: float = 1.0

    @property
    def cell(self) -> Cell:
        return (self.col, self.row)

    @property
    def identified(self) -> bool:
        return self.piece is not None

    @property
    def part(self) -> PieceRef | None:
        return None if self.piece is None else PieceRef(piece=self.piece)

    @property
    def pose(self) -> CellPose:
        return CellPose(col=self.col, row=self.row, rot=self.rot)  # type: ignore[arg-type]


@dataclass
class PieceGridState(WorldState):
    """What is on the board, and what changed since the last reading."""

    cols: int
    rows: int
    #: About half a second (FR-STA-3).
    stable_s: float = 0.5
    #: Below this a reading is not considered at all; the checker's alert
    #: threshold (FR-CHK-9) downgrades instead of dropping.
    min_confidence: float = 0.25

    placed: dict[Cell, PieceReading] = field(default_factory=dict, init=False)
    #: Cells holding a piece we could not identify.
    unidentified: set[Cell] = field(default_factory=set, init=False)
    _seen: dict[Cell, tuple[PieceReading, float]] = field(default_factory=dict, init=False)
    _gone: dict[Cell, float] = field(default_factory=dict, init=False)
    _pending: list[Event] = field(default_factory=list, init=False)
    occluded: bool = field(default=False, init=False)
    #: Readings dropped for being off-board or unconfident.
    dropped: int = field(default=0, init=False)
    #: Count of readings left unidentified, for the perception report.
    unidentified_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._t = 0.0
        self._reverify: set[str] = set()
        self._reverify_clear = False

    # --- reading the board ---------------------------------------------------------

    def observe(self, t: float, readings: list[PieceReading], hands_present: bool = False) -> None:
        """Take one frame's readings. Call every frame; committing happens here."""
        self._t = t
        self.occluded = hands_present
        if hands_present:
            self._seen.clear()
            self._gone.clear()
            return

        usable: dict[Cell, PieceReading] = {}
        for r in readings:
            if (
                r.confidence < self.min_confidence
                or not (0 <= r.col < self.cols and 0 <= r.row < self.rows)
            ):
                self.dropped += 1
                continue
            usable[r.cell] = r
        identified = {c: r for c, r in usable.items() if r.identified}
        self.unidentified = {c for c, r in usable.items() if not r.identified}
        self.unidentified_seen += len(self.unidentified)

        for cell, r in identified.items():
            previous = self._seen.get(cell)
            if previous is not None and (previous[0].piece, previous[0].rot) == (r.piece, r.rot):
                self._seen[cell] = (r, previous[1])
            else:
                self._seen[cell] = (r, t)
        for cell in set(self._seen) - set(identified):
            del self._seen[cell]

        for cell in self.placed:
            # Present as something we cannot name still counts as present.
            if cell in usable:
                self._gone.pop(cell, None)
            else:
                self._gone.setdefault(cell, t)

        self._commit(t)
        self._reverify_clear = True

    def _commit(self, t: float) -> None:
        for cell, (r, since) in sorted(self._seen.items()):
            if t - since < self.stable_s:
                continue
            current = self.placed.get(cell)
            if current is not None and (current.piece, current.rot) == (r.piece, r.rot):
                continue
            if current is not None:
                # A different piece or turn in a settled cell: report both
                # halves so the checker reopens the step before judging.
                self._pending.append(self._event(t, "REMOVE", current))
            self.placed[cell] = r
            self._pending.append(self._event(t, "ADD", r))

        for cell, since in list(self._gone.items()):
            if t - since < self.stable_s:
                continue
            del self._gone[cell]
            if cell in self.placed:
                self._pending.append(self._event(t, "REMOVE", self.placed.pop(cell)))

    @staticmethod
    def _event(t: float, kind: str, r: PieceReading) -> Event:
        return Event(
            t=t,
            kind=kind,  # type: ignore[arg-type]
            part=r.part,
            pose=r.pose,
            confidence=r.confidence,
        )

    # --- WorldState ----------------------------------------------------------------

    def current(self) -> dict[Cell, PieceReading | None]:
        """Every occupied cell: its reading, or None where the piece is unidentified."""
        out: dict[Cell, PieceReading | None] = {c: r for c, r in self.placed.items()}
        out.update({c: None for c in self.unidentified if c not in out})
        return out

    def diff(self) -> list[Event]:
        events, self._pending = self._pending, []
        return events

    def predicates(self):
        """A piece grid supplies no lattice predicates; DAG preconditions use `done()` only."""
        return {}, {}

    # --- re-verification (FR-STA-6) ------------------------------------------------

    def reverify_ready(self) -> bool:
        return super().reverify_ready() and not self.occluded

    def confirmed_at(self, part: PieceRef, pose: CellPose) -> bool:
        """Is this exact piece at this cell and rotation right now?"""
        cell = (pose.col, pose.row)
        seen = self._seen.get(cell)
        candidates = [self.placed.get(cell), None if seen is None else seen[0]]
        return any(
            r is not None and r.piece == part.piece and r.rot == pose.rot for r in candidates
        )
