"""Spec validation (FR-CMP-6, FR-CMP-7).

A compiled spec is rejected unless it passes every check below. The output is a
list of `Problem`s rather than an exception, because FR-CMP-7 needs the failing
check and the offending id to hand back to the LLM for one repair round, and
FR-CMP-9 needs them to show a human in the spec editor.

Errors block the run. Warnings do not, but they are the compiler-accuracy
signal that EV-CMP-2 reports on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from thirdeye.compiler.expr import check_expr
from thirdeye.compiler.schema import ConstraintSpec, DagSpec, JigsawSpec, Step

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Problem:
    code: str
    severity: Severity
    target: str  # step, action or precondition id ("" when spec-wide)
    message: str

    def __str__(self) -> str:
        where = f" [{self.target}]" if self.target else ""
        return f"{self.severity}: {self.code}{where}: {self.message}"


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


def is_valid(problems: list[Problem]) -> bool:
    return not errors(problems)


# --- DAG checks ---------------------------------------------------------------------

def _check_ids(spec: DagSpec | JigsawSpec) -> list[Problem]:
    seen: set[str] = set()
    out = []
    for step in spec.steps:
        if step.id in seen:
            out.append(Problem("duplicate_id", "error", step.id, "step id is used twice"))
        seen.add(step.id)
    for step in spec.steps:
        for kind in ("hard_depends_on", "soft_depends_on"):
            for dep in getattr(step, kind):
                if dep not in seen:
                    out.append(
                        Problem("unknown_dependency", "error", step.id,
                                f"{kind} names {dep!r}, which is not a step")
                    )
                elif dep == step.id:
                    out.append(
                        Problem("self_dependency", "error", step.id, "step depends on itself")
                    )
    return out


def _check_cycles(spec: DagSpec | JigsawSpec) -> list[Problem]:
    """Depth-first cycle detection over hard and soft dependencies together."""
    ids = {s.id for s in spec.steps}
    deps = {
        s.id: [d for d in (*s.hard_depends_on, *s.soft_depends_on) if d in ids and d != s.id]
        for s in spec.steps
    }
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(deps, WHITE)
    out: list[Problem] = []

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GREY
        for dep in deps[node]:
            if colour[dep] == GREY:
                cycle = " -> ".join([*path[path.index(dep):], node, dep])
                out.append(Problem("cycle", "error", node, f"dependency cycle: {cycle}"))
            elif colour[dep] == WHITE:
                visit(dep, [*path, node])
        colour[node] = BLACK

    for node in deps:
        if colour[node] == WHITE:
            visit(node, [])
    return out


def _check_bounds(spec: DagSpec) -> list[Problem]:
    out = []
    for step in spec.steps:
        outside = {
            (x, y) for x, y in step.footprint
            if not (0 <= x < spec.baseplate.studs_x and 0 <= y < spec.baseplate.studs_y)
        }
        if outside:
            out.append(
                Problem("out_of_bounds", "error", step.id,
                        f"{step.part} at {step.pose.x},{step.pose.y} runs off the baseplate "
                        f"at {sorted(outside)[:3]}")
            )
    return out


def _check_collisions(spec: DagSpec) -> list[Problem]:
    """Two parts may not occupy the same cell on the same layer."""
    occupied: dict[tuple[int, int, int], str] = {}
    out = []
    for step in spec.steps:
        for x, y in step.footprint:
            key = (x, y, step.pose.layer)
            if key in occupied:
                out.append(
                    Problem("collision", "error", step.id,
                            f"cell {x},{y} on layer {step.pose.layer} is already used "
                            f"by {occupied[key]}")
                )
            else:
                occupied[key] = step.id
    return out


def _supporters(step: Step, spec: DagSpec) -> set[str]:
    """Steps whose footprint lies directly beneath this one."""
    if step.pose.layer == 0:
        return set()
    below = step.pose.layer - 1
    cells = step.footprint
    return {
        other.id for other in spec.steps
        if other.id != step.id and other.pose.layer == below and other.footprint & cells
    }


def _check_support(spec: DagSpec) -> list[Problem]:
    """Nothing floats, and hard dependencies match the geometry that implies them."""
    out = []
    for step in spec.steps:
        if step.pose.layer == 0:
            continue
        supporters = _supporters(step, spec)
        if not supporters:
            out.append(
                Problem("floating", "error", step.id,
                        f"sits on layer {step.pose.layer} with nothing beneath it")
            )
            continue
        # The compiler is meant to record physical support as a hard dependency
        # (FR-CMP-3). A gap here is the single most useful compiler-accuracy
        # signal we have, and it is exactly what EV-CMP-2 scores.
        missing = supporters - set(step.hard_depends_on)
        if missing:
            out.append(
                Problem("missing_hard_dependency", "warning", step.id,
                        f"physically supported by {sorted(missing)}, "
                        f"but they are not in hard_depends_on")
            )
    return out


def _check_soft_order(spec: DagSpec | JigsawSpec) -> list[Problem]:
    """A hard dependency that is not also a soft one is almost always a slip."""
    return [
        Problem("hard_dep_not_ordered", "warning", step.id,
                f"{sorted(set(step.hard_depends_on) - set(step.soft_depends_on))} "
                f"is a hard dependency but not a stated order")
        for step in spec.steps
        if set(step.hard_depends_on) - set(step.soft_depends_on)
    ]


# --- jigsaw checks ------------------------------------------------------------------

def _check_cells(spec: JigsawSpec) -> list[Problem]:
    """Every target cell lies on the grid (FR-CMP-6's bounds check, for a flat grid)."""
    return [
        Problem("out_of_bounds", "error", step.id,
                f"cell (col {step.pose.col}, row {step.pose.row}) is outside the "
                f"{spec.grid.cols}x{spec.grid.rows} grid")
        for step in spec.steps
        if not (step.pose.col < spec.grid.cols and step.pose.row < spec.grid.rows)
    ]


def _check_cell_collisions(spec: JigsawSpec) -> list[Problem]:
    """Two pieces may not be assigned the same cell, nor one piece two cells."""
    cells: dict[tuple[int, int], str] = {}
    pieces: dict[str, str] = {}
    out = []
    for step in spec.steps:
        key = (step.pose.col, step.pose.row)
        if key in cells:
            out.append(Problem("collision", "error", step.id,
                               f"cell {key} is already used by {cells[key]}"))
        else:
            cells[key] = step.id
        if step.part.piece in pieces:
            out.append(Problem("duplicate_piece", "error", step.id,
                               f"piece {step.part.piece} is already placed by {pieces[step.part.piece]}"))
        else:
            pieces[step.part.piece] = step.id
    return out


def _check_hard_deps(spec: JigsawSpec) -> list[Problem]:
    """Nothing holds a flat puzzle piece up, so a hard dependency is suspect."""
    return [
        Problem("unexpected_hard_dependency", "warning", step.id,
                f"hard_depends_on {step.hard_depends_on} on a flat jigsaw; order should be soft")
        for step in spec.steps
        if step.hard_depends_on
    ]


# --- constraint checks --------------------------------------------------------------

def _check_exprs(spec: ConstraintSpec) -> list[Problem]:
    out = []
    for action in spec.actions:
        for pre in action.preconditions:
            out += [
                Problem("bad_expr", "error", pre.id, err.message)
                for err in check_expr(pre.expr)
            ]
    for term in spec.terminal:
        out += [
            Problem("bad_expr", "error", term.id, err.message)
            for err in check_expr(term.expr)
        ]
    return out


def _check_constraint_ids(spec: ConstraintSpec) -> list[Problem]:
    seen: set[str] = set()
    out = []
    for action in spec.actions:
        if action.id in seen:
            out.append(Problem("duplicate_id", "error", action.id, "action id is used twice"))
        seen.add(action.id)
        pre_ids: set[str] = set()
        for pre in action.preconditions:
            if pre.id in pre_ids:
                out.append(
                    Problem("duplicate_id", "error", pre.id, "precondition id is used twice")
                )
            pre_ids.add(pre.id)
    return out


def _check_satisfiable(spec: ConstraintSpec) -> list[Problem]:
    """At least one action, a way for a game to end, and a legal first move.

    The last is FR-CMP-6's satisfiability check proper: a rulebook compiled so
    strictly that nothing is legal from the opening position would flag every
    move of every game. It is found by trying every move from the initial state
    against every alerting precondition, one hand on the tower.
    """
    out = []
    if not spec.actions:
        out.append(Problem("unsatisfiable", "error", "", "spec declares no actions"))
    for action in spec.actions:
        if not action.preconditions:
            out.append(
                Problem("unconstrained_action", "warning", action.id,
                        "action has no preconditions, so nothing about it can be illegal")
            )
    if not spec.terminal:
        out.append(
            Problem("no_terminal", "warning", "",
                    "no terminal condition, so a session can never end on its own")
        )
    if spec.actions and spec.state_model == "tower_lattice" and not _has_legal_opening(spec):
        out.append(
            Problem("unsatisfiable", "error", "",
                    "no move is legal from the initial tower, so every move would be a foul")
        )
    return out


def _has_legal_opening(spec: ConstraintSpec) -> bool:
    # Imported here: the validator needs a state model to simulate against, but
    # nothing else in the compiler should depend on one.
    from thirdeye.checker.evaluate import Environment, EvalError, evaluate
    from thirdeye.events import Slot
    from thirdeye.state.jenga.lattice import TowerLattice, full_tower

    layers, slots = spec.tower.layers, spec.tower.slots_per_layer
    lattice = TowerLattice(initial=full_tower(layers, slots), slots_per_layer=slots)
    base = Environment(funcs={"done": lambda _id: False}).merge(*lattice.predicates())
    cells = [Slot(layer, s) for layer in range(layers + 1) for s in range(slots)]
    for action in spec.actions:
        pres = [p for p in action.preconditions if p.on_violation == "alert"]
        for src in cells[: layers * slots]:
            for dst in cells:
                env = base.merge({"src": src, "dst": dst, "hands_in_contact": 1})
                try:
                    if all(evaluate(p.expr, env) for p in pres):
                        return True
                except EvalError:
                    continue
    return False


# --- entry point --------------------------------------------------------------------

def validate(spec: DagSpec | JigsawSpec | ConstraintSpec) -> list[Problem]:
    """Every problem with this spec, errors and warnings, in a stable order."""
    if isinstance(spec, JigsawSpec):
        problems = (
            _check_ids(spec)
            + _check_cells(spec)
            + _check_cell_collisions(spec)
            + _check_hard_deps(spec)
            + _check_soft_order(spec)
        )
        if not [p for p in problems if p.code in {"unknown_dependency", "self_dependency"}]:
            problems += _check_cycles(spec)
    elif isinstance(spec, DagSpec):
        problems = (
            _check_ids(spec)
            + _check_bounds(spec)
            + _check_collisions(spec)
            + _check_support(spec)
            + _check_soft_order(spec)
        )
        # A cycle makes the traversal meaningless, so only look once ids resolve.
        if not [p for p in problems if p.code in {"unknown_dependency", "self_dependency"}]:
            problems += _check_cycles(spec)
    else:
        problems = _check_constraint_ids(spec) + _check_exprs(spec) + _check_satisfiable(spec)

    order = {"error": 0, "warning": 1}
    return sorted(problems, key=lambda p: (order[p.severity], p.target, p.code))
