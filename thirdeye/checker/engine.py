"""The checker (FR-CHK-1, FR-CHK-2, FR-CHK-7, FR-CHK-11, FR-CHK-12).

The engine judges events against a lowered spec and nothing else. It does not
know whether the procedure is LEGO or Jenga, ordered or unordered: `spec.lower()`
hands it one list of `LoweredAction`, and every judgement it makes comes out of
the same loop over that action's preconditions. This is FR-CHK-2 cashed in, and
it is the design claim the report makes.

Two things the engine deliberately does not do:

*Simulate.* A spec's `effects` are recorded, never applied. The world state is
whatever the cameras last confirmed (see `state/base.py`), so the only effect an
event has here is on step *status*.

*Guess.* When a precondition cannot be evaluated, or a hard dependency looks
unmet, the engine declines to decide rather than alerting. An alert is a claim
about the user, and we do not make one from a failure of our own (FR-CHK-7).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field

from thirdeye.checker.evaluate import Environment, EvalError, evaluate, missing_names
from thirdeye.checker.match import MatchPolicy, policy_for
from thirdeye.compiler.schema import ConstraintSpec, DagSpec, LoweredAction
from thirdeye.events import (
    COMPLETED,
    OPEN,
    Alert,
    Event,
    LogEntry,
    LogKind,
    Reverification,
    Severity,
    Status,
)

#: Names an event supplies rather than the state model. They are always
#: considered available at setup; whether a given event carries them is a
#: per-event question, answered by declining to judge rather than by alerting.
EVENT_NAMES = frozenset({"src", "dst", "hands_in_contact"})

MISSED_STEP = "missed_step"
EXTRA_PART = "extra_part"


class SpecNotRunnable(Exception):
    """This spec cannot be judged against this state model (IF-SPEC-1).

    Raised at construction, never mid-session. A Jenga rulebook handed a stud
    grid fails here, loudly, instead of producing an unexplained silence during
    a demo.
    """


@dataclass
class _Deferred:
    """A judgement postponed pending re-verification (FR-CHK-7)."""

    event: Event
    action: LoweredAction
    deps: set[str]


@dataclass
class Checker:
    """Judges a session's events against one procedure spec.

    Construct once per session. `on_event` is on the per-event path and must
    stay within NFR-PERF-8 (1 ms); it allocates no more than the alerts it
    returns.
    """

    spec: DagSpec | ConstraintSpec
    state: object | None = None
    #: FR-CHK-9. An event we are not confident about can only produce a yellow
    #: warning, whatever rule it broke.
    confidence_threshold: float = 0.5
    #: Injectable for tests; otherwise selected from the spec's state model.
    match_policy: InitVar[MatchPolicy | None] = None

    status: dict[str, Status] = field(default_factory=dict, init=False)
    alerts: list[Alert] = field(default_factory=list, init=False)
    log: list[LogEntry] = field(default_factory=list, init=False)
    #: Steps a re-read proved were there all along. Ours, not the user's; they
    #: are counted against detector recall in the evaluation (FR-CHK-8).
    perception_misses: list[str] = field(default_factory=list, init=False)
    #: Precondition ids we could not evaluate, so the report can state which
    #: rules were not actually in force during a session.
    unevaluable: list[str] = field(default_factory=list, init=False)
    outcome: str | None = field(default=None, init=False)

    policy: MatchPolicy = field(init=False)

    def __post_init__(self, match_policy: MatchPolicy | None) -> None:
        self.policy = match_policy or policy_for(self.spec)
        self.actions = self.spec.lower()
        self.status = {a.id: "pending" for a in self.actions}
        self._by_id = {a.id: a for a in self.actions}
        self._deferred: list[_Deferred] = []
        #: Steps already reported missing by a re-read. Their status stays
        #: `pending` -- the brick really is absent, so `done()` must stay false
        #: for anything depending on it -- so the session-end sweep needs this
        #: to avoid alerting on the same absence twice.
        self._reported_missing: set[str] = set()
        self._t = 0.0
        unrunnable = self._unrunnable()
        if unrunnable:
            raise SpecNotRunnable(
                f"{self.spec.procedure} reads state this session cannot observe: "
                + "; ".join(unrunnable)
            )

    # --- setup ---------------------------------------------------------------------

    def _state_env(self) -> Environment:
        env = Environment(funcs={"done": self._done})
        if self.state is not None and hasattr(self.state, "predicates"):
            names, funcs = self.state.predicates()
            env = env.merge(names, funcs)
        return env

    def _unrunnable(self) -> list[str]:
        """Expressions in this spec that read something we cannot supply.

        Checked once, here, so that FR-CHK-2's single code path cannot turn a
        state-model mismatch into a mid-session surprise.
        """
        available = set(self._state_env().bound()) | set(EVENT_NAMES)
        out = []
        for action in self.actions:
            for pre in action.preconditions:
                gap = missing_names(pre.expr, available)
                if gap:
                    out.append(f"{pre.id} needs {', '.join(gap)}")
        for term in getattr(self.spec, "terminal", []):
            gap = missing_names(term.expr, available)
            if gap:
                out.append(f"{term.id} needs {', '.join(gap)}")
        return out

    # --- the event path ------------------------------------------------------------

    def on_event(self, ev: Event) -> list[Alert]:
        """Judge one event. Returns only the alerts this event produced."""
        self._t = max(self._t, ev.t)
        self._note(ev.t, "event", ev.describe())
        if self.outcome is not None:
            return []

        if ev.kind == "COLLAPSE":
            return self._terminate(ev.t, "collapse", "game_over")
        if ev.kind == "STEP_START":
            return self._on_step_hint(ev)
        if ev.kind == "STEP_END":
            return []

        if ev.kind == "REMOVE":
            raised = self.policy.on_remove(ev, self.actions, self.status)
            return self._raise_all(ev, raised)

        raised = self._on_action_event(ev)
        raised += self._check_terminal(ev)
        return raised

    def _on_action_event(self, ev: Event) -> list[Alert]:
        action = self.policy.match(ev, self.actions, self.status)
        if action is None:
            return self._raise_all(ev, self.policy.unmatched(ev, self.actions, self.status))

        # Hard dependencies first. They are physically necessary, so an unmet
        # one means we misread the scene, and the judgement must wait for a
        # clear look rather than accuse the user (FR-CHK-7).
        unmet_hard = self._failing(action.deferring(), ev)
        if unmet_hard:
            deps = {self._dep_of(p.id) for p in unmet_hard}
            self._deferred.append(_Deferred(ev, action, deps))
            if self.state is not None and hasattr(self.state, "request_reverify"):
                self.state.request_reverify(deps)
            self._note(ev.t, "deferred", f"awaiting re-read of {sorted(deps)}", action.id)
            return []

        return self._judge(ev, action)

    def _judge(self, ev: Event, action: LoweredAction) -> list[Alert]:
        """Run the alerting preconditions and record the action as performed."""
        failing = self._failing(action.alerting(), ev)
        raised = self._raise_all(
            ev,
            [
                Alert(
                    t=ev.t,
                    code=pre.violation,
                    rule=pre.id,
                    text=self.policy.explain(pre, action, ev),
                    target=action.id,
                )
                for pre in failing
            ],
        )
        self._apply(ev, action, errored=bool(raised))
        return raised

    def _failing(self, pres: Iterable, ev: Event) -> list:
        """The preconditions that are false and that we could actually evaluate.

        A precondition we cannot evaluate is skipped, logged, and counted -- not
        treated as satisfied and not treated as violated. Two cheaper designs
        are both wrong: alerting would accuse the user of a gap in our own
        perception (FR-CHK-7), and abandoning the whole judgement would let one
        missing signal switch off every unrelated rule on the same action. A
        Jenga move with an unknown hand count must still be checked for an
        illegal source layer.

        The skips are accumulated in `unevaluable` so the session report can say
        which rules were not actually in force (FR-CHK-8's honesty, applied to
        rules rather than to detections).
        """
        env = self._event_env(ev)
        out = []
        for pre in pres:
            try:
                if not evaluate(pre.expr, env):
                    out.append(pre)
            except EvalError as exc:
                self._note(ev.t, "internal", f"cannot evaluate {pre.expr!r}: {exc}", pre.id)
                self.unevaluable.append(pre.id)
        return out

    def _event_env(self, ev: Event) -> Environment:
        return self._state_env().merge(
            {"src": ev.src, "dst": ev.dst, "hands_in_contact": ev.hands_in_contact}
        )

    def _apply(self, ev: Event, action: LoweredAction, errored: bool = False) -> None:
        """Record that `action` happened.

        Effects are logged, not executed: the state comes from the cameras, and
        a checker that wrote to it would be marking its own homework.
        """
        if action.effects:
            self._note(ev.t, "status", f"declared effects {action.effects}", action.id)
        if action.repeatable:
            return
        # `error` rather than `done` so the checklist can show it red, but both
        # count as performed for `done()`; see `COMPLETED`.
        self.status[action.id] = "error" if errored else "done"
        self._note(ev.t, "status", f"-> {self.status[action.id]}", action.id)

    def _on_step_hint(self, ev: Event) -> list[Alert]:
        """A temporal-model step hint marks a step under way, never done.

        The temporal model is the only signal available while hands occlude the
        workspace (plan 2.5), but FR-CHK-11 forbids an alert that rests on a
        learned model alone, so a hint can move a step to `active` and no
        further. Confirmation comes from the state diff.
        """
        for action in self.actions:
            if self.status.get(action.id) == "pending" and self.policy.matches_hint(ev, action):
                self.status[action.id] = "active"
                self._note(ev.t, "status", "-> active (temporal hint)", action.id)
        return []

    # --- re-verification (FR-CHK-7, FR-CHK-8) --------------------------------------

    def on_reverify(self, rev: Reverification) -> list[Alert]:
        """Resolve deferred judgements now that the view is clear.

        A dependency the second look confirms was a miss of ours, logged against
        detector recall. One it cannot find was genuinely skipped, and *that* is
        a missed step the user hears about.
        """
        self._t = max(self._t, rev.t)
        asked = {d for deferred in self._deferred for d in deferred.deps}
        raised: list[Alert] = []
        for dep in sorted(asked):
            if self.status.get(dep) in COMPLETED:
                # The tracker caught up on its own before the re-read landed --
                # a late commit, not a miss. Counting it would understate our
                # detector's recall as surely as ignoring a real miss overstates it.
                self._note(rev.t, "resolved", "confirmed by the tracker before the re-read", dep)
            elif rev.confirms(dep):
                self.status[dep] = "done"
                self.perception_misses.append(dep)
                self._note(rev.t, "perception_miss", "present all along; we missed it", dep)
            else:
                self._reported_missing.add(dep)
                raised += self._raise_all(
                    None,
                    [
                        Alert(
                            t=rev.t,
                            code=MISSED_STEP,
                            rule="checker.missed_step",
                            text=self.policy.missed_text(self._by_id.get(dep), dep),
                            target=dep,
                        )
                    ],
                )

        # Resume the postponed judgements. The placement itself was observed, so
        # the step completes either way -- anything else would report it a second
        # time as a missed step at session end.
        still: list[_Deferred] = []
        for deferred in self._deferred:
            if deferred.deps <= asked:
                self._note(rev.t, "resolved", "deferred judgement resumed", deferred.action.id)
                raised += self._judge(deferred.event, deferred.action)
            else:
                still.append(deferred)
        self._deferred = still
        return raised

    # --- terminal conditions and session end ---------------------------------------

    def _check_terminal(self, ev: Event) -> list[Alert]:
        env = self._event_env(ev)
        for term in getattr(self.spec, "terminal", []):
            try:
                if evaluate(term.expr, env):
                    return self._terminate(ev.t, term.id, term.outcome)
            except EvalError as exc:
                self._note(ev.t, "internal", f"cannot evaluate {term.expr!r}: {exc}", term.id)
        return []

    def _terminate(self, t: float, rule: str, outcome: str) -> list[Alert]:
        self.outcome = outcome
        return self._raise_all(
            None,
            [
                Alert(
                    t=t,
                    code=outcome,
                    rule=rule,
                    text=f"Session ended: {outcome.replace('_', ' ')}.",
                )
            ],
        )

    def finish(self, t: float | None = None) -> list[Alert]:
        """Close the session: every required step still pending is a miss (FR-CHK-3)."""
        t = self._t if t is None else t
        raised: list[Alert] = []

        # A deferral never answered is our unfinished business. The placement
        # was observed, so the step completes -- but its alerting preconditions
        # are *not* run. We deferred precisely because we doubted our reading of
        # the dependency, and judging against that same doubtful reading is the
        # thing FR-CHK-7 forbids. The absent dependency is still reported by the
        # sweep below, so the user gets the one alert that is actually warranted.
        for deferred in self._deferred:
            self._note(t, "internal", "re-read never arrived; completing unjudged",
                       deferred.action.id)
            self._apply(deferred.event, deferred.action)
        self._deferred = []

        for action in self.actions:
            if action.optional or action.repeatable:
                continue
            if self.status[action.id] not in OPEN:
                continue
            if action.id in self._reported_missing:
                continue
            raised += self._raise_all(
                None,
                [
                    Alert(
                        t=t,
                        code=MISSED_STEP,
                        rule="checker.missed_step",
                        text=self.policy.missed_text(action, action.id),
                        target=action.id,
                    )
                ],
            )
        return raised

    # --- bookkeeping ---------------------------------------------------------------

    def _done(self, step_id: str) -> bool:
        """The rulebook's `done(id)`: is the part there?

        True for `error` as well as `done`. A step performed out of order was
        still performed, and the alert already said so -- see `COMPLETED`.
        """
        return self.status.get(step_id) in COMPLETED

    @staticmethod
    def _dep_of(precondition_id: str) -> str:
        """`s2.hard.s1` -> `s1`. The id shape is set by `DagSpec.lower`."""
        return precondition_id.rsplit(".", 1)[-1]

    def _severity(self, ev: Event | None) -> Severity:
        """FR-CHK-9: below the confidence threshold nothing goes red."""
        if ev is None:
            return "alert"
        return "alert" if ev.confidence >= self.confidence_threshold else "warning"

    def _raise_all(self, ev: Event | None, alerts: list[Alert]) -> list[Alert]:
        out = []
        for alert in alerts:
            graded = Alert(
                t=alert.t,
                code=alert.code,
                rule=alert.rule,
                text=alert.text,
                severity=self._severity(ev),
                target=alert.target,
            )
            self.alerts.append(graded)
            self.log.append(LogEntry(graded.t, "alert", str(graded), graded.target))
            out.append(graded)
        return out

    def _note(self, t: float, kind: LogKind, detail: str, target: str = "") -> None:
        self.log.append(LogEntry(t, kind, detail, target))

    # --- views for the UI ----------------------------------------------------------

    def action(self, action_id: str) -> LoweredAction | None:
        """The lowered action behind an id -- how a caller resolves a step id to
        a part and a pose without reading the spec's surface form."""
        return self._by_id.get(action_id)

    def checklist(self) -> list[tuple[str, Status, str]]:
        """(id, status, instruction) in spec order, for the live checklist."""
        return [(a.id, self.status[a.id], a.instruction) for a in self.actions]

    def next_expected(self) -> LoweredAction | None:
        """The first step whose dependencies are all met -- the ghost overlay."""
        for action in self.actions:
            if action.repeatable or self.status[action.id] != "pending":
                continue
            if not self._failing_ids(action):
                return action
        return None

    def _failing_ids(self, action: LoweredAction) -> list[str]:
        env = self._state_env()
        out = []
        for pre in action.preconditions:
            try:
                if not evaluate(pre.expr, env):
                    out.append(pre.id)
            except EvalError:
                pass  # needs an event to evaluate; not a blocker for this view
        return out
