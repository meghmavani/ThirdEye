"""Safe precondition expressions (FR-CMP-5).

Preconditions arrive from an LLM, so they are never executed as Python. An
expression is parsed with `ast`, every node is checked against a whitelist, and
every name it mentions must come from the fixed predicate vocabulary below --
the same vocabulary the compiler prompt hands the model.

This module only *checks* expressions. Evaluation against live state is the
checker's job (FR-CHK-1) and lives in thirdeye/checker/.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# --- the fixed predicate vocabulary -------------------------------------------------
# Anything not named here is a compiler error, not a runtime surprise.

#: Bare names the checker binds before evaluating an expression.
STATE_VARS: dict[str, str] = {
    "src": "the slot or pose a part is taken from; has .layer and .slot",
    "dst": "the slot or pose a part is placed at; has .layer and .slot",
    "hands_in_contact": "how many hands are touching the workspace right now",
    "top_layer": "index of the highest layer holding any part",
    "top_complete_layer": "index of the highest layer with every slot filled",
    "collapse_detected": "true on the frame a collapse is observed",
}

#: Attributes reachable on a state variable.
ATTRS: dict[str, tuple[str, ...]] = {
    "src": ("layer", "slot", "x", "y", "rot"),
    "dst": ("layer", "slot", "x", "y", "rot"),
}

#: Callables the checker exposes to expressions.
PREDICATES: dict[str, str] = {
    "done": "done('s3') -- has that step been completed",
    "layer_complete": "layer_complete(n) -- is every slot in layer n filled",
    "legal_top_slots": "legal_top_slots() -- the slots a part may legally be placed in",
    "occupancy": "occupancy(layer, slot) -- 1 if filled, 0 if empty",
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.BinOp, ast.Add, ast.Sub,           # for "top_complete_layer - 1"
    ast.Call, ast.Name, ast.Attribute, ast.Constant, ast.Load,
)


@dataclass(frozen=True)
class ExprError:
    """One reason an expression was rejected."""

    expr: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.message} in {self.expr!r}"


def vocabulary_prompt() -> str:
    """The vocabulary block to paste into the compiler prompt (FR-CMP-5)."""
    lines = ["State variables:"]
    lines += [f"  {name} -- {desc}" for name, desc in STATE_VARS.items()]
    lines.append("Attributes:")
    lines += [f"  {var}.{{{', '.join(attrs)}}}" for var, attrs in ATTRS.items()]
    lines.append("Functions:")
    lines += [f"  {desc}" for desc in PREDICATES.values()]
    lines.append("Operators: and or not == != < <= > >= in, + and - on integers.")
    lines.append("Nothing else is permitted. Do not invent predicates.")
    return "\n".join(lines)


def check_expr(expr: str) -> list[ExprError]:
    """Return every reason `expr` is not a legal precondition. Empty means legal."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return [ExprError(expr, f"does not parse ({exc.msg})")]

    errors: list[ExprError] = []
    # Names used as call targets are checked as predicates, not as state variables.
    call_names = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            errors.append(ExprError(expr, f"{type(node).__name__} is not allowed"))
            continue

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                errors.append(ExprError(expr, "only plain function calls are allowed"))
            elif node.func.id not in PREDICATES:
                errors.append(ExprError(expr, f"unknown predicate {node.func.id!r}"))
            if node.keywords:
                errors.append(ExprError(expr, "keyword arguments are not allowed"))

        elif isinstance(node, ast.Attribute):
            root = node.value
            if not isinstance(root, ast.Name):
                errors.append(ExprError(expr, "attributes are only allowed on state variables"))
            elif node.attr not in ATTRS.get(root.id, ()):
                errors.append(ExprError(expr, f"unknown attribute {root.id}.{node.attr}"))

        elif isinstance(node, ast.Name):
            if node.id not in STATE_VARS and node.id not in call_names:
                errors.append(ExprError(expr, f"unknown name {node.id!r}"))

    # Deduplicate while keeping order -- ast.walk can reach the same name twice.
    seen: set[str] = set()
    unique = []
    for err in errors:
        if err.message not in seen:
            seen.add(err.message)
            unique.append(err)
    return unique
