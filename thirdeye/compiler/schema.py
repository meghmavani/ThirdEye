"""Procedure spec schemas (FR-CMP-1..4, IF-SPEC-1..5).

Two surface forms, one checker interface:

    DAG form         -- ordered steps with dependencies (LEGO, plan 4.2)
    constraint form  -- actions with preconditions, no order (Jenga, plan 4.3)

The checker is constraint-native (FR-CHK-2): both forms `lower()` to the same
list of `LoweredAction`, so no engine code branches on procedure type. The DAG
is the special case -- step k's precondition is "everything it depends on is
done" -- and lowering is where that claim is actually cashed in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

Rotation = Literal[0, 90, 180, 270]

#: A hard-dependency violation is physically impossible, so observing one means
#: our perception is wrong, not that the user erred. It must never reach the
#: user as an alert (FR-CHK-7) -- it routes to re-verification instead.
HARD_DEP_VIOLATION = "hard_dependency_unmet"


class Strict(BaseModel):
    """Reject unknown fields: an LLM inventing a key is a compile error, not a shrug."""

    model_config = ConfigDict(extra="forbid")


# --- parts and poses ----------------------------------------------------------------

class PartRef(Strict):
    """A part by the only two properties the detector reports (FR-PER-2)."""

    color: str
    size: str  # "2x4"

    @field_validator("size")
    @classmethod
    def _size_form(cls, v: str) -> str:
        if not re.fullmatch(r"\d+x\d+", v):
            raise ValueError(f"size must look like '2x4', got {v!r}")
        return v

    @property
    def dims(self) -> tuple[int, int]:
        w, h = self.size.split("x")
        return int(w), int(h)

    def __str__(self) -> str:
        return f"{self.color}_{self.size}"


class Pose(Strict):
    """A placement on the stud grid. `x`, `y` are the low corner in studs."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    layer: int = Field(default=0, ge=0)
    rot: Rotation = 0

    def footprint(self, part: PartRef) -> set[tuple[int, int]]:
        """The stud cells this part occupies at this pose, rotation applied."""
        w, h = part.dims
        if self.rot in (90, 270):
            w, h = h, w
        return {(self.x + dx, self.y + dy) for dx in range(w) for dy in range(h)}


# --- provenance (IF-SPEC-5) ---------------------------------------------------------

class Provenance(Strict):
    """Who produced a spec. Stamped by the compiler, never written by the LLM --
    the schema the model is shown leaves this field out."""

    compiler_version: str
    #: The LLM that produced it, or "hand" / "demo" for the other two routes.
    model: str
    source: str = ""
    compiled_at: str = ""
    repaired: bool = False


# --- DAG form (plan 4.2) ------------------------------------------------------------

class Step(Strict):
    id: str
    instruction: str
    part: PartRef
    pose: Pose
    #: Physically required support. Violation => our detection failed (FR-CMP-3).
    hard_depends_on: list[str] = Field(default_factory=list)
    #: Order stated by the manual. Violation => a real out-of-order error.
    soft_depends_on: list[str] = Field(default_factory=list)
    optional: bool = False
    expected_duration_s: tuple[float, float] = (3.0, 20.0)

    @field_validator("expected_duration_s")
    @classmethod
    def _duration_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo < 0 or hi < lo:
            raise ValueError(f"expected_duration_s must be [lo, hi] with 0 <= lo <= hi, got {v}")
        return v

    @property
    def footprint(self) -> set[tuple[int, int]]:
        return self.pose.footprint(self.part)


class Baseplate(Strict):
    studs_x: int = Field(gt=0)
    studs_y: int = Field(gt=0)


class DagSpec(Strict):
    procedure: str
    kind: Literal["dag"] = "dag"
    state_model: Literal["stud_grid"] = "stud_grid"
    baseplate: Baseplate
    steps: list[Step]
    provenance: Provenance | None = None

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def lower(self) -> list[LoweredAction]:
        """Generate constraint-form preconditions from the dependency lists.

        Hard and soft dependencies produce the same predicate -- `done(dep)` --
        and differ only in what happens when it fails, which is exactly the
        distinction FR-CHK-7 turns on.
        """
        actions = []
        for step in self.steps:
            pre = [
                Precondition(
                    id=f"{step.id}.hard.{dep}",
                    expr=f"done({dep!r})",
                    text=f"{dep} must be placed first to physically support this part",
                    violation=HARD_DEP_VIOLATION,
                    on_violation="reverify",
                )
                for dep in step.hard_depends_on
            ]
            pre += [
                Precondition(
                    id=f"{step.id}.soft.{dep}",
                    expr=f"done({dep!r})",
                    text=f"the manual places {dep} before this step",
                    violation="out_of_order",
                    on_violation="alert",
                )
                for dep in step.soft_depends_on
            ]
            actions.append(
                LoweredAction(
                    id=step.id,
                    instruction=step.instruction,
                    part=step.part,
                    pose=step.pose,
                    optional=step.optional,
                    repeatable=False,
                    preconditions=pre,
                    effects=[f"place({step.id})"],
                )
            )
        return actions


# --- constraint form (plan 4.3) -----------------------------------------------------

class Precondition(Strict):
    id: str
    #: Checked against the fixed vocabulary in expr.py; never executed as Python.
    expr: str
    #: Human-readable rule text -- what the user is shown (NFR-USE-2).
    text: str
    #: Machine code logged with the alert (FR-CHK-11).
    violation: str
    #: "alert" reaches the user; "reverify" never does (FR-CHK-7).
    on_violation: Literal["alert", "reverify"] = "alert"


class Action(Strict):
    id: str
    instruction: str
    preconditions: list[Precondition]
    effects: list[str] = Field(default_factory=list)


class Terminal(Strict):
    id: str
    expr: str
    outcome: str


class Tower(Strict):
    layers: int = Field(gt=0)
    slots_per_layer: int = Field(gt=0)


class ConstraintSpec(Strict):
    procedure: str
    kind: Literal["constraints"] = "constraints"
    state_model: Literal["tower_lattice"] = "tower_lattice"
    tower: Tower
    actions: list[Action]
    terminal: list[Terminal] = Field(default_factory=list)
    provenance: Provenance | None = None

    def lower(self) -> list[LoweredAction]:
        """Already native; carried across unchanged."""
        return [
            LoweredAction(
                id=a.id,
                instruction=a.instruction,
                part=None,
                pose=None,
                optional=False,
                repeatable=True,
                preconditions=list(a.preconditions),
                effects=list(a.effects),
            )
            for a in self.actions
        ]


# --- what the checker actually consumes ---------------------------------------------

class LoweredAction(Strict):
    """The single form the checker engine sees (FR-CHK-2).

    `part` and `pose` are set for placement-shaped procedures and None for
    procedures whose actions are not tied to a part identity (Jenga's one
    block class), so the engine matches on pose only when there is one.
    """

    id: str
    instruction: str
    part: PartRef | None = None
    pose: Pose | None = None
    optional: bool = False
    #: A DAG step is done once and never judged again. A constraint action --
    #: Jenga's `move` -- governs every matching event for the whole session.
    #: This is the only thing that differs between the two forms downstream,
    #: and it is a field rather than a branch in the engine (FR-CHK-2).
    repeatable: bool = False
    preconditions: list[Precondition] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)

    def alerting(self) -> list[Precondition]:
        return [p for p in self.preconditions if p.on_violation == "alert"]

    def deferring(self) -> list[Precondition]:
        return [p for p in self.preconditions if p.on_violation == "reverify"]


ProcedureSpec = Annotated[DagSpec | ConstraintSpec, Field(discriminator="kind")]


_ADAPTER: TypeAdapter[DagSpec | ConstraintSpec] = TypeAdapter(ProcedureSpec)


def parse_spec(data: dict) -> DagSpec | ConstraintSpec:
    """Parse either form, dispatching on `kind` (IF-SPEC-1)."""
    return _ADAPTER.validate_python(data)


def load_spec(path: str | Path) -> DagSpec | ConstraintSpec:
    """Load a spec file. A spec that does not validate does not run (IF-SPEC-1)."""
    return parse_spec(json.loads(Path(path).read_text()))
