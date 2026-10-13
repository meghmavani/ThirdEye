"""Matching and alert text for the tower lattice (FR-CHK-10).

Every Jenga rule lives in the compiled rulebook, not here: illegal source
layer, one hand at a time, finish a layer before starting the next, and legal
replacement slots are all preconditions in the spec (plan 4.3), evaluated by
the engine's generic loop. That is the tier-3 result -- the *spec format* is
general, not just the weights -- so a rule implemented in this file would
quietly undo it.

What is left is the two things a rulebook cannot state: pairing the halves of a
move into one judgeable event, and phrasing a violation in terms of the slots
actually involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thirdeye.compiler.schema import LoweredAction, Precondition
from thirdeye.events import Alert, Event, Status


class TowerLatticePolicy:
    """The Jenga half of the checker: thin, on purpose."""

    def match(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> LoweredAction | None:
        if ev.kind != "MOVE":
            return None
        # A rulebook compiles to action schemas with no part identity. One
        # schema means the match is unambiguous; a rulebook with several
        # distinct actions would need a discriminator on the event, and the
        # Jenga rules do not have one.
        return next((a for a in actions if a.part is None and a.pose is None), None)

    def unmatched(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """Nothing. There is one block class and no pose to get wrong, so an
        unmatched event is a gap in our state, not a foul by the player (plan
        1.1) -- and FR-CHK-7's principle forbids billing that to the user."""
        return []

    def on_remove(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """Nothing on its own: a removal is half a move. `MovePairer` joins it
        to the placement that follows and the engine judges the pair."""
        return []

    def matches_hint(self, ev: Event, action: LoweredAction) -> bool:
        return False

    def explain(self, pre: Precondition, action: LoweredAction, ev: Event) -> str:
        where = ""
        if ev.src is not None or ev.dst is not None:
            where = f" (took from {ev.src or 'somewhere'}, placed at {ev.dst or 'somewhere'})"
        return f"{pre.text}{where}."

    def missed_text(self, action: LoweredAction | None, action_id: str) -> str:
        # Unreachable in practice: constraint actions are repeatable, so the
        # session-end sweep skips them. A rulebook has no steps to miss.
        return f"{action_id} was never performed."


@dataclass
class MovePairer:
    """Joins a `REMOVE` and the `ADD` that completes it into one `MOVE`.

    The lattice tracker reports slot changes (FR-STA-8), but a Jenga rule is
    about a move: `src.layer` and `dst.layer` are read by the same precondition.
    Judging the removal alone would mean deciding the legality of a placement
    that has not happened yet.

    Hand count is taken from the removal, because the rule the rulebook states
    -- one hand at a time -- is about the pull.
    """

    #: How long a removed block may stay in hand before we stop expecting it
    #: back. Generous: players deliberate, and a dropped pair is worse than a
    #: late one.
    window_s: float = 20.0

    held: Event | None = field(default=None, init=False)
    #: Removals that timed out unpaired. Logged, never alerted -- losing track
    #: of a block is our failure.
    dropped: list[Event] = field(default_factory=list, init=False)

    def feed(self, ev: Event) -> list[Event]:
        """Events to hand the checker, given this observation."""
        self._expire(ev.t)

        if ev.kind == "REMOVE" and ev.src is not None:
            if self.held is not None:
                self.dropped.append(self.held)
            self.held = ev
            return []

        if ev.kind == "ADD" and ev.dst is not None and self.held is not None:
            removal, self.held = self.held, None
            return [
                Event(
                    t=ev.t,
                    kind="MOVE",
                    source=ev.source,
                    src=removal.src,
                    dst=ev.dst,
                    hands_in_contact=removal.hands_in_contact,
                    confidence=min(removal.confidence, ev.confidence),
                )
            ]

        # An arrival with nothing in hand, or anything else (COLLAPSE), passes
        # straight through. The lattice policy matches only MOVE, so a stray
        # ADD is silently ignored rather than blamed on the player.
        return [ev]

    def _expire(self, now: float) -> None:
        if self.held is not None and now - self.held.t > self.window_s:
            self.dropped.append(self.held)
            self.held = None
