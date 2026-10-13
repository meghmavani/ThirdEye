"""The machine-readable session log (IF-LOG-1, FR-CHK-12, FR-EXP-2).

One JSON file per session, holding everything the alert timeline and the LLM
report are regenerated from -- and nothing that needs the video. The explainer
reads this file, never the session object, so FR-EXP-2 ("from the event log
only") is true by construction rather than by discipline.

The file is a record of judgements, not a replay of perception: it keeps the
events the checker saw and what it made of them. Our own failures travel in
their own fields (`perception_misses`, `unevaluable`), apart from the alerts, so
no downstream consumer can add them up into one number by accident (FR-CHK-8).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from thirdeye.events import Alert, LogEntry

#: Bumped on any change a reader of an older file would misread (IF-WS-2's
#: versioning rule, applied to the log for the same reason).
FORMAT_VERSION = 1


@dataclass
class ChecklistRow:
    id: str
    status: str
    instruction: str


@dataclass
class SessionRecord:
    """A finished session, as the evaluation and the explainer see it."""

    procedure: str
    kind: str
    alerts: list[Alert]
    log: list[LogEntry]
    checklist: list[ChecklistRow]
    perception_misses: list[str] = field(default_factory=list)
    unevaluable: list[str] = field(default_factory=list)
    outcome: str | None = None
    #: Free-form provenance: script path, config, git commit (NFR-REP-1).
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = FORMAT_VERSION

    @classmethod
    def of(cls, session, meta: dict[str, Any] | None = None) -> SessionRecord:
        """Snapshot a `Session`. Taken after `finish()`, it is the whole story."""
        return cls(
            procedure=session.spec.procedure,
            kind=session.spec.kind,
            alerts=list(session.alerts),
            log=list(session.log),
            checklist=[ChecklistRow(*row) for row in session.checklist()],
            perception_misses=sorted(set(session.checker.perception_misses)),
            unevaluable=sorted(set(session.checker.unevaluable)),
            outcome=session.outcome,
            meta=dict(meta or {}),
        )

    # --- views -------------------------------------------------------------------

    def red(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == "alert"]

    def yellow(self) -> list[Alert]:
        return [a for a in self.alerts if a.severity == "warning"]

    def timeline(self) -> str:
        """The alert timeline, one line per alert, as shown at the end of a demo."""
        return "\n".join(str(a) for a in self.alerts)

    # --- files -------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SessionRecord:
        version = data.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"session log is format {version}, this reader understands {FORMAT_VERSION}"
            )
        return cls(
            procedure=data["procedure"],
            kind=data["kind"],
            alerts=[Alert(**a) for a in data["alerts"]],
            log=[LogEntry(**e) for e in data["log"]],
            checklist=[ChecklistRow(**r) for r in data["checklist"]],
            perception_misses=list(data.get("perception_misses", [])),
            unevaluable=list(data.get("unevaluable", [])),
            outcome=data.get("outcome"),
            meta=dict(data.get("meta", {})),
            version=version,
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n")
        return path

    @classmethod
    def read(cls, path: str | Path) -> SessionRecord:
        return cls.from_json(json.loads(Path(path).read_text()))
