"""The LEGO build-state tracker (FR-STA-2, FR-STA-3, FR-STA-4, FR-STA-6).

Holds the model as placed bricks in stud coordinates and turns changes between
readings into `ADD` and `REMOVE` events. Two things guard it against the failure
mode that dominates this workspace -- a hand over the plate:

*Freeze while occluded* (FR-PER-6). A reading taken with a hand across the
baseplate is not a weak reading, it is a wrong one: bricks behind the hand simply
are not there as far as the detector is concerned. So nothing is committed while
hands are present, and the stability timers restart when they leave.

*Commit only what has held still* (FR-STA-3). Each cell has its own timer, in
both directions: a brick must be seen continuously for `stable_s` before it is
placed, and missing continuously for `stable_s` before it is removed. Per-cell
rather than whole-scene, so one flickering detection in a corner cannot stall
every other commit.

The tracker knows nothing about steps or about the procedure. It reports what is
on the plate; the checker decides whether that is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thirdeye.compiler.schema import PartRef, Pose
from thirdeye.events import Event
from thirdeye.state.base import WorldState
from thirdeye.state.lego.grid import Reading, StudGrid, pose_matches

Cell = tuple[int, int, int]


@dataclass
class BuildState(WorldState):
    """What is on the baseplate, and what changed since the last reading."""

    grid: StudGrid
    #: About half a second at 15-30 FPS (FR-STA-3).
    stable_s: float = 0.5
    #: Below this, a detection is not considered at all. Distinct from the
    #: checker's alert threshold (FR-CHK-9), which downgrades rather than drops.
    min_confidence: float = 0.25
    #: A detection whose centroid sits further than this from the cell centre is
    #: not really on the grid. One stud is the p95 the evaluation targets.
    max_residual: float = 1.0

    placed: dict[Cell, Reading] = field(default_factory=dict, init=False)
    #: Readings we have been seeing, and since when.
    _seen: dict[Cell, tuple[Reading, float]] = field(default_factory=dict, init=False)
    #: Committed cells currently not seen, and since when.
    _gone: dict[Cell, float] = field(default_factory=dict, init=False)
    _pending: list[Event] = field(default_factory=list, init=False)
    #: True while a hand is over the workspace. Set from the last reading.
    occluded: bool = field(default=False, init=False)
    #: Detections dropped for being off-grid or unconfident, for the perception
    #: report -- a silently ignored detection is a debugging trap.
    dropped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._t = 0.0
        self._reverify: set[str] = set()
        self._reverify_clear = False

    # --- reading the plate ----------------------------------------------------------

    def observe(self, t: float, readings: list[Reading], hands_present: bool = False) -> None:
        """Take one frame's detections.

        Call every frame. Committing happens here, so `diff()` is a pure read
        and stays well inside the 1 ms of NFR-PERF-7.
        """
        self._t = t
        self.occluded = hands_present
        if hands_present:
            # Frozen. Timers restart once the view is clear, so a brick is not
            # credited for time it spent behind a hand.
            self._seen.clear()
            self._gone.clear()
            return

        usable = {}
        for reading in readings:
            if reading.confidence < self.min_confidence or reading.residual > self.max_residual:
                self.dropped += 1
                continue
            usable[reading.cell] = reading

        # A cell whose part changed identity restarts its timer: that is a
        # different brick, not the same one settling.
        for cell, reading in usable.items():
            previous = self._seen.get(cell)
            if previous is not None and previous[0].part == reading.part:
                self._seen[cell] = (reading, previous[1])
            else:
                self._seen[cell] = (reading, t)
        for cell in set(self._seen) - set(usable):
            del self._seen[cell]

        for cell in self.placed:
            if cell in usable:
                self._gone.pop(cell, None)
            else:
                self._gone.setdefault(cell, t)

        self._commit(t)
        # A clean, committed reading is exactly what the deferred decision was
        # waiting for.
        self._reverify_clear = True

    def _commit(self, t: float) -> None:
        # Bottom layer first. Bricks settling in the same frame were stacked in
        # that order, and the checker would otherwise see a brick before the one
        # holding it up and defer a judgement over nothing.
        for cell, (reading, since) in sorted(self._seen.items(), key=lambda kv: kv[0][2]):
            if t - since < self.stable_s:
                continue
            current = self.placed.get(cell)
            if current is not None and current.part == reading.part:
                continue
            if current is not None:
                # A different brick in a cell we thought was settled: report the
                # swap as both halves, so the checker sees the removal it needs
                # to reopen the step before judging the new placement.
                self._pending.append(self._event(t, "REMOVE", current))
            self.placed[cell] = reading
            self._pending.append(self._event(t, "ADD", reading))

        for cell, since in list(self._gone.items()):
            if t - since < self.stable_s:
                continue
            del self._gone[cell]
            if cell in self.placed:
                self._pending.append(self._event(t, "REMOVE", self.placed.pop(cell)))

    @staticmethod
    def _event(t: float, kind: str, reading: Reading) -> Event:
        return Event(
            t=t,
            kind=kind,  # type: ignore[arg-type]
            part=reading.part,
            pose=reading.pose,
            confidence=reading.confidence,
        )

    # --- WorldState -----------------------------------------------------------------

    def current(self) -> tuple[Reading, ...]:
        """The placed bricks, in a stable order (FR-STA-2)."""
        return tuple(self.placed[cell] for cell in sorted(self.placed))

    def diff(self) -> list[Event]:
        events, self._pending = self._pending, []
        return events

    def predicates(self):
        """A stud grid supplies no lattice predicates.

        Deliberately empty. A Jenga rulebook run against this tracker is refused
        at session setup rather than failing mid-run, because the checker checks
        what the state can answer before it accepts the spec.
        """
        return {}, {}

    # --- re-verification (FR-STA-6) -------------------------------------------------

    def reverify_ready(self) -> bool:
        """A stud grid knows when its view is obstructed, so it can say more
        than the base class: not merely that a reading has happened, but that a
        hand was not across the plate during it."""
        return super().reverify_ready() and not self.occluded

    def confirmed_at(self, part: PartRef, pose: Pose) -> bool:
        """Is this exact part at this pose right now?

        The question the re-verification path asks of each deferred dependency.
        A brick in the latest clear reading counts even if it has not yet held
        still for `stable_s`: the stability filter guards blind commits, but a
        re-read is a targeted look for one known part in one known cell, and
        making it wait out the timer would report a brick that is sitting right
        there as a missed step.
        """
        cell = (pose.x, pose.y, pose.layer)
        seen = self._seen.get(cell)
        candidates = [self.placed.get(cell), None if seen is None else seen[0]]
        return any(
            r is not None and r.part == part and pose_matches(r.pose, pose, part)
            for r in candidates
        )
