"""The WebSocket message schema, versioned (IF-WS-1, IF-WS-2).

Everything the UI shows is one of these messages, sent **outbound only**: the browser
never sends video or state back during a demo (IF-WS-1). A UI and a backend can be
built against this file alone.

    {"v": 1, "type": "hello",  "procedures": [...]}
    {"v": 1, "type": "status", "running": bool, "procedure": str, "source": str}
    {"v": 1, "type": "state",  "t": float, "procedure": str, "kind": str,
        "checklist": [{"id","status","instruction"}],     # empty for a rulebook
        "next": {"id","instruction"} | null,               # the ghost (FR-UI-5)
        "legality": {...} | null,                          # Jenga's legal-moves panel
        "overlay": {"model": "piece_grid"|"tower_lattice"|"stud_grid", ...},
        "outcome": str | null,
        "counts": {"alerts": int, "warnings": int}}
    {"v": 1, "type": "alert",  "t","code","rule","text","severity","target"}
    {"v": 1, "type": "done",   "summary": str, "report": str}

`overlay` is what the camera view would draw (FR-UI-1): the lattice or the piece grid
with what is in each cell. Bump `PROTOCOL_VERSION` on any change a client built for
the old schema would misread.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from thirdeye.events import Alert
from thirdeye.session import Session
from thirdeye.state.jenga.lattice import (
    TowerLattice,
    legal_top_slots,
    top_complete_layer,
    top_layer,
)
from thirdeye.state.jigsaw.piece_grid import PieceGridState
from thirdeye.state.lego.build_state import BuildState

PROTOCOL_VERSION = 1


def msg(type_: str, /, **fields: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": type_, **fields}


def alert_message(alert: Alert) -> dict[str, Any]:
    return msg("alert", **asdict(alert))


def overlay(session: Session) -> dict[str, Any]:
    state = session.state
    if isinstance(state, TowerLattice):
        snap = state.current()
        return {"model": "tower_lattice", "rows": [list(r) for r in snap]}
    if isinstance(state, PieceGridState):
        cells = [
            {"col": c, "row": r, "piece": None if reading is None else reading.piece,
             "rot": 0 if reading is None else reading.rot}
            for (c, r), reading in sorted(state.current().items())
        ]
        return {"model": "piece_grid", "cols": state.cols, "rows": state.rows, "cells": cells}
    if isinstance(state, BuildState):
        return {"model": "stud_grid", "bricks": [
            {"part": str(b.part), "x": b.pose.x, "y": b.pose.y, "layer": b.pose.layer,
             "rot": b.pose.rot} for b in state.current()]}
    return {"model": "unknown"}  # pragma: no cover - a new state model adds a branch


def legality(session: Session) -> dict[str, Any] | None:
    """Jenga's live legality panel (FR-UI-2): what a player may do right now."""
    state = session.state
    if not isinstance(state, TowerLattice):
        return None
    snap = state.before()
    tcl = top_complete_layer(snap)
    return {
        "top_layer": top_layer(snap),
        "top_complete_layer": tcl,
        # Layers a block may be taken from, by the rulebook's headline rule.
        "takeable_up_to_layer": tcl - 1,
        "legal_slots": [str(s) for s in sorted(legal_top_slots(snap))],
        "rules": [
            {"id": p.id, "text": p.text}
            for a in session.checker.actions for p in a.alerting()
        ],
    }


def state_message(session: Session, t: float) -> dict[str, Any]:
    nxt = session.checker.next_expected()
    red = sum(1 for a in session.alerts if a.severity == "alert")
    rulebook = all(a.repeatable for a in session.checker.actions)
    return msg(
        "state",
        t=t,
        procedure=session.spec.procedure,
        kind=session.spec.kind,
        checklist=[] if rulebook else [
            {"id": i, "status": s, "instruction": text} for i, s, text in session.checklist()
        ],
        next=None if nxt is None else {"id": nxt.id, "instruction": nxt.instruction},
        legality=legality(session),
        overlay=overlay(session),
        outcome=session.outcome,
        counts={"alerts": red, "warnings": len(session.alerts) - red},
    )
