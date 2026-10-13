"""The session report (FR-EXP-1, FR-EXP-2, EV-BL-2's counterpart on the output side).

    python -m thirdeye.explain.report game.json            # deterministic report, no LLM
    python -m thirdeye.explain.report game.json --claude   # the LLM writes it

Takes a `SessionRecord` and nothing else, so "from the event log only, never from
video" (FR-EXP-2) is true by construction: there is no other argument to pass. The
report turns the alert timeline into prose -- "At 03:10 you pulled from layer 16, one
below the top complete layer -- illegal" -- and, for a collapse, reconstructs the last
few moves.

Two ways to write it, one input:

*`plain_report`* is deterministic text built straight from the record. It needs no
network and no key, so a session is always explainable offline (FR-SYS-4), and it is
what the LLM version is checked against: the model may rephrase, but every claim in
its report must be traceable to a line of this one.

*`explain`* sends the record to an LLM as text (FR-CMP-11's rule, for the explainer)
with instructions to cite only what is in the log. It is called once, after the
session, never on the per-frame or per-event path.

Our own failures are reported apart from the user's. Perception misses and rules that
were not in force are *our* errors (FR-CHK-8); the prompt says so, and the plain report
puts them in their own section, so neither can be read as something the player did.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol

from thirdeye.sessionlog import SessionRecord

#: How many moves before a collapse the report walks back through.
COLLAPSE_LOOKBACK = 5


class ReportLLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


def clock(t: float) -> str:
    """Seconds as mm:ss, the way a person reads a timeline."""
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def last_moves(record: SessionRecord, n: int = COLLAPSE_LOOKBACK) -> list[str]:
    """The last `n` observed events before the session ended, newest last."""
    events = [e for e in record.log if e.kind == "event" and not e.detail.startswith("COLLAPSE")]
    return [f"{clock(e.t)} {e.detail}" for e in events[-n:]]


def timeline(record: SessionRecord) -> list[str]:
    return [
        f"{clock(a.t)} [{'ALERT' if a.severity == 'alert' else 'warning'}] {a.code} "
        f"(rule {a.rule}{', ' + a.target if a.target else ''}): {a.text}"
        for a in record.alerts
    ]


def plain_report(record: SessionRecord) -> str:
    """The report without a model: every line is a fact from the record."""
    lines = [f"Session report: {record.procedure}", ""]
    red, yellow = record.red(), record.yellow()
    done = [r for r in record.checklist if r.status == "done"]
    if record.kind == "dag":
        lines.append(f"Progress: {len(done)} of {len(record.checklist)} steps completed.")
    outcome = {"game_over": "The game ended: the tower fell."}.get(
        record.outcome or "", f"The session ended: {record.outcome}." if record.outcome else ""
    )
    if outcome:
        lines.append(outcome)
    lines.append(f"{len(red)} alert(s) and {len(yellow)} warning(s) were raised.")
    lines.append("")

    if record.alerts:
        lines.append("What happened:")
        lines += [f"  - At {line}" for line in timeline(record)]
    else:
        lines.append("No rule was broken.")

    if record.outcome and record.kind == "constraints":
        moves = last_moves(record)
        if moves:
            lines += ["", "The last moves before the end:"] + [f"  - {m}" for m in moves]

    pending = [r for r in record.checklist if r.status in ("pending", "active")]
    if pending and record.kind == "dag":
        lines += ["", "Never completed:"] + [f"  - {r.id}: {r.instruction}" for r in pending]

    if record.perception_misses or record.unevaluable:
        lines += ["", "On our side, not yours:"]
        if record.perception_misses:
            lines.append("  - The camera missed these steps at first, though they were done: "
                         + ", ".join(record.perception_misses))
        if record.unevaluable:
            lines.append("  - These rules could not be checked this session: "
                         + ", ".join(record.unevaluable))
    return "\n".join(lines) + "\n"


_SYSTEM = """\
You write the post-game / post-build report for a system that watches someone follow a
procedure (a Jenga game or a jigsaw) and flags rule violations. You are given the
session log as text. Write a short, plain-language report for the person who took part.

Rules you must follow:
- Use only facts in the log. Quote times as mm:ss exactly as given. Never invent a
  move, a time, a rule or a consequence that is not in the log.
- For each alert, say what happened, when, which rule it broke (use the rule text),
  and the consequence the log supports. Group repeats.
- If the session ended in a collapse, say so and walk back through the last moves listed.
- "On our side" items (perception misses, rules not in force) are failures of the
  system, not the player. Mention them once, apologetically, under their own heading,
  and never attribute them to the player.
- Yellow warnings are suspicions, not violations: describe them as such.
- Be concise. No headings beyond: Summary, What happened, On our side (only if present).
"""


def build_prompt(record: SessionRecord) -> str:
    parts = [
        f"Procedure: {record.procedure} ({record.kind} form)",
        f"Outcome: {record.outcome or 'finished without a terminal event'}",
        "",
        "Checklist:" if record.checklist else "Checklist: (none; a rulebook has no steps)",
        *[f"  {r.id} [{r.status}] {r.instruction}" for r in record.checklist],
        "",
        "Alert timeline:" if record.alerts else "Alert timeline: (no alerts)",
        *[f"  {line}" for line in timeline(record)],
    ]
    moves = last_moves(record)
    if record.outcome and moves:
        parts += ["", "Last events before the end:", *[f"  {m}" for m in moves]]
    if record.perception_misses:
        parts += ["", "ON OUR SIDE - steps the camera missed at first: "
                  + ", ".join(record.perception_misses)]
    if record.unevaluable:
        parts += ["ON OUR SIDE - rules that could not be checked: " + ", ".join(record.unevaluable)]
    return "\n".join(parts)


def explain(record: SessionRecord, llm: ReportLLM | None = None) -> str:
    """The LLM's report, or the plain one when no model is given."""
    if llm is None:
        return plain_report(record)
    return llm.complete(_SYSTEM, build_prompt(record))


class ClaudeReporter:
    """Claude through the Anthropic SDK (the `llm` extra). Key from the environment (NFR-PRIV-2)."""

    def __init__(self, model: str | None = None, max_tokens: int = 2000, client=None):
        from thirdeye.compiler.compile import DEFAULT_MODEL

        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = client

    def complete(self, system: str, user: str) -> str:
        if self.client is None:
            import anthropic  # the `llm` extra, imported where used

            self.client = anthropic.Anthropic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path, help="a session log written by --log (IF-LOG-1)")
    ap.add_argument("--claude", action="store_true", help="have Claude write the report")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    record = SessionRecord.read(args.log)
    print(explain(record, ClaudeReporter(args.model) if args.claude else None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
