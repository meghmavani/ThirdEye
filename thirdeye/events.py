"""The event and alert vocabulary shared by the state trackers and the checker.

Plan section 4.4 sketches an event as a dict. It is a frozen dataclass here for
one reason: the checker binds event fields straight into precondition
expressions (`src.layer`, `hands_in_contact`), and a typo in a dict key would
surface as an unbound name at event time instead of a load-time error.

State trackers produce these (FR-STA-4, FR-STA-8); the checker consumes them
(FR-CHK-1). Nothing here knows which procedure is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from thirdeye.compiler.schema import PartRef, Pose

#: FR-CHK-1. `active` is set when a step is under way but not yet confirmed --
#: it exists so the UI can distinguish "not started" from "in progress".
Status = Literal["pending", "active", "done", "error"]

#: Statuses meaning "the part is there". An out-of-order step is `error` --
#: flagged for the UI -- but it was still performed, so `done()` must be true of
#: it; so is a step attempted with the wrong brick (see `rules_lego`). Treating a flagged step as not done would cascade one mistake into an
#: out-of-order alert on every step that depends on it, against the budget of
#: one false alert per build (NFR-REL-2).
COMPLETED: tuple[Status, ...] = ("done", "error")

#: Statuses meaning "not performed yet", and so the ones a session-end sweep
#: reports as missed (FR-CHK-3).
OPEN: tuple[Status, ...] = ("pending", "active")

#: FR-CHK-9. A red alert is a claim; a yellow warning is a suspicion. Events
#: below the confidence threshold can only ever produce the latter.
Severity = Literal["alert", "warning"]

EventKind = Literal["ADD", "REMOVE", "MOVE", "STEP_START", "STEP_END", "COLLAPSE"]

EventSource = Literal["state", "temporal"]


@dataclass(frozen=True, order=True)
class Slot:
    """A Jenga lattice cell: which layer, which of the three slots across it."""

    layer: int
    slot: int

    def __str__(self) -> str:
        return f"L{self.layer}s{self.slot}"


@dataclass(frozen=True)
class Event:
    """One observation the checker must judge (plan 4.4).

    Placement-shaped procedures fill `part` and `pose`; a Jenga move fills
    `src` and `dst`. The checker never asks which kind it has -- it binds
    whatever is present and a spec that reads a field this event does not carry
    is rejected when the session is set up, not when the event arrives.
    """

    t: float
    kind: EventKind
    source: EventSource = "state"
    part: PartRef | None = None
    pose: Pose | None = None
    src: Slot | None = None
    dst: Slot | None = None
    #: A Jenga rule (FR-CHK-10) reads this directly, so it travels on the event
    #: rather than being fetched from perception after the fact.
    hands_in_contact: int | None = None
    confidence: float = 1.0

    def describe(self) -> str:
        where = str(self.pose_str() or self.slots_str() or "")
        what = f" {self.part}" if self.part else ""
        return f"{self.kind}{what}{' @ ' + where if where else ''}"

    def pose_str(self) -> str | None:
        p = self.pose
        return None if p is None else f"({p.x},{p.y},L{p.layer},rot{p.rot})"

    def slots_str(self) -> str | None:
        if self.src is None and self.dst is None:
            return None
        return f"{self.src or '-'}->{self.dst or '-'}"


@dataclass(frozen=True)
class Alert:
    """Something the user is told.

    Every alert names the rule that produced it (FR-CHK-11), so no alert can
    come from a learned model's opinion alone, and carries plain-language text
    naming the step and the consequence (NFR-USE-2). `code` is the stable
    machine label the evaluation scripts group by.
    """

    t: float
    code: str
    rule: str
    text: str
    severity: Severity = "alert"
    target: str = ""

    def __str__(self) -> str:
        mark = "!" if self.severity == "alert" else "?"
        return f"[{self.t:7.2f}] {mark} {self.code}: {self.text}"


#: What a log line is about. `perception_miss` and `internal` are ours, not the
#: user's, and are counted against detector recall instead (FR-CHK-8).
LogKind = Literal[
    "event", "alert", "deferred", "resolved", "perception_miss", "status", "internal"
]


@dataclass(frozen=True)
class LogEntry:
    """One line of the session log (FR-CHK-12)."""

    t: float
    kind: LogKind
    detail: str
    target: str = ""

    def __str__(self) -> str:
        where = f" [{self.target}]" if self.target else ""
        return f"[{self.t:7.2f}] {self.kind}{where}: {self.detail}"


@dataclass
class Reverification:
    """The answer to a `request_reverify` (FR-STA-6).

    `supported` names the steps a second, unobstructed look confirmed were
    present all along -- we missed them. Everything else asked about was
    genuinely absent.
    """

    t: float
    supported: set[str] = field(default_factory=set)

    def confirms(self, step_id: str) -> bool:
        return step_id in self.supported
