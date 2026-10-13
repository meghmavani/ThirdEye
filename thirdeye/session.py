"""One run of one procedure: the join between the tracker and the checker.

The online path of plan section 4.5, minus perception. A `Session` owns a state
model and a checker and does the two things neither can do alone:

*Drains events.* The tracker reports what changed; the checker judges it.

*Closes the re-verification loop* (FR-CHK-7, FR-STA-6). The checker defers a
judgement and asks for a second look; the tracker says when its view is clear;
and deciding *which* steps that second look confirmed needs the spec, which
only the session has. This is the one place the three meet, and keeping it here
is why the tracker stays reusable for another procedure (NFR-MNT-1) and the
checker stays free of grid geometry.

*Assembles moves.* A lattice reports slot changes, but a Jenga rule judges a
move; the pairer that joins the halves sits between the two (see
`rules_jenga.MovePairer`).

Perception plugs in above: call the tracker's own `observe` with each frame's
detections, then `pump` to get the alerts. Nothing here touches a camera, so a
whole session can be replayed from a script -- see `thirdeye/replay.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thirdeye.checker.engine import Checker
from thirdeye.checker.rules_jenga import MovePairer
from thirdeye.compiler.schema import ConstraintSpec, DagSpec, JigsawSpec
from thirdeye.events import Alert, Event, LogEntry, Status
from thirdeye.state.base import WorldState
from thirdeye.state.jigsaw.piece_grid import PieceGridState
from thirdeye.state.jenga.lattice import TowerLattice
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import StudGrid


def state_for(spec: DagSpec | JigsawSpec | ConstraintSpec) -> WorldState:
    """Build the state model the spec declares.

    Keyed on `state_model`, the same axis `policy_for` uses, so a new state
    model is registered in two places and changes no existing code.
    """
    if spec.state_model == "stud_grid":
        return BuildState(StudGrid.from_spec(spec.baseplate))
    if spec.state_model == "piece_grid":
        return PieceGridState(cols=spec.grid.cols, rows=spec.grid.rows)
    if spec.state_model == "tower_lattice":
        return TowerLattice(slots_per_layer=spec.tower.slots_per_layer)
    raise ValueError(f"no state model named {spec.state_model!r}")  # pragma: no cover


def pairer_for(spec: DagSpec | JigsawSpec | ConstraintSpec) -> MovePairer | None:
    """The event assembler this state model needs between tracker and checker.

    Only a lattice needs one: its tracker reports the halves of a move, and its
    rules judge the whole. A stud grid's ADD is already a complete action.
    """
    return MovePairer() if spec.state_model == "tower_lattice" else None


@dataclass
class Session:
    """A procedure being performed, and the judgement of it so far."""

    spec: DagSpec | JigsawSpec | ConstraintSpec
    state: WorldState = field(default=None)  # type: ignore[assignment]
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = state_for(self.spec)
        self.checker = Checker(
            self.spec, state=self.state, confidence_threshold=self.confidence_threshold
        )
        self.pairer = pairer_for(self.spec)

    # --- the online loop ------------------------------------------------------------

    def pump(self) -> list[Alert]:
        """Judge everything the tracker has reported since the last call.

        Re-verification is resolved *before* the new events, never after. A
        deferral raised by this batch cannot be answered from the reading that
        raised it: that reading is exactly the one we doubt, and re-reading it
        would reproduce the mistake. The tracker invalidates it when asked, and
        the answer comes from the next clear reading, on a later pump.
        """
        alerts = self._resolve_reverification()
        for event in self.state.diff():
            alerts += self._judge(event)
        return alerts

    def submit(self, event: Event) -> list[Alert]:
        """Judge one event directly, bypassing the tracker.

        For signals the state model does not produce: a step hint from the
        temporal model, or a hand count from perception.
        """
        return self._resolve_reverification() + self._judge(event)

    def _judge(self, event: Event) -> list[Alert]:
        events = [event] if self.pairer is None else self.pairer.feed(event)
        alerts: list[Alert] = []
        for ev in events:
            alerts += self.checker.on_event(ev)
        return alerts

    def _resolve_reverification(self) -> list[Alert]:
        pending = self.state.pending_reverify()
        if not pending or not self.state.reverify_ready():
            return []
        confirmed = {step_id for step_id in pending if self._observed(step_id)}
        answer = self.state.take_reverification(confirmed)
        return [] if answer is None else self.checker.on_reverify(answer)

    def _observed(self, step_id: str) -> bool:
        """Is the part this step places on the plate right now?

        Resolved through the checker's lowered actions rather than the spec's
        surface form, so this works for either procedure kind. A state model
        that cannot answer the question reports nothing rather than guessing --
        and a guess of `True` would silently convert a real skipped step into a
        logged perception miss, flattering our own recall.
        """
        action = self.checker.action(step_id)
        confirmed_at = getattr(self.state, "confirmed_at", None)
        if action is None or action.part is None or action.pose is None or confirmed_at is None:
            return False
        return bool(confirmed_at(action.part, action.pose))

    def finish(self, t: float | None = None) -> list[Alert]:
        """Close the session and sweep up whatever was never done."""
        return self.checker.finish(t)

    # --- views ----------------------------------------------------------------------

    @property
    def alerts(self) -> list[Alert]:
        return self.checker.alerts

    @property
    def log(self) -> list[LogEntry]:
        return self.checker.log

    @property
    def outcome(self) -> str | None:
        return self.checker.outcome

    def checklist(self) -> list[tuple[str, Status, str]]:
        return self.checker.checklist()

    def summary(self) -> str:
        """A few lines for a terminal or the end of a demo.

        Perception misses are printed separately and *not* as alerts: they are
        our detector's recall, not the user's mistakes (FR-CHK-8), and mixing
        them into one number is how a project ends up reporting a score it did
        not earn.
        """
        red = sum(1 for a in self.alerts if a.severity == "alert")
        yellow = len(self.alerts) - red
        if all(a.repeatable for a in self.checker.actions):
            # A rulebook has no steps to tick off; what it governs is moves.
            moves = sum(1 for e in self.log if e.kind == "event" and e.detail.startswith("MOVE"))
            progress = f"{moves} moves judged"
        else:
            done = sum(1 for _, status, _ in self.checklist() if status == "done")
            progress = f"{done}/{len(self.checklist())} steps done"
        headline = f"{self.spec.procedure}: {progress}, {red} alerts, {yellow} warnings"
        lines = [headline]
        if self.checker.perception_misses:
            lines.append(
                f"  our misses (not the user's): {sorted(set(self.checker.perception_misses))}"
            )
        if self.checker.unevaluable:
            lines.append(f"  rules not in force: {sorted(set(self.checker.unevaluable))}")
        if self.outcome:
            lines.append(f"  ended: {self.outcome}")
        return "\n".join(lines)
