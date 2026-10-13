"""Evaluating precondition expressions against observed state (FR-CMP-5, FR-CHK-1).

`compiler/expr.py` decides whether an expression is *legal*. This module decides
whether it is *true right now*. The split matters: legality is checked once, at
compile time, and is the thing FR-CMP-5 requires; truth is checked per event and
must run inside the 1 ms budget of NFR-PERF-8.

There is no `eval` here and no code path that reaches one. An expression is
parsed to an AST, re-checked against the whitelist as a second line of defence,
and then walked by an explicit visitor that knows how to compute exactly the
node types `expr.py` permits. A node the visitor does not recognise raises
rather than falling through to anything.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from thirdeye.compiler.expr import ATTRS, PREDICATES, STATE_VARS, check_expr


class EvalError(Exception):
    """An expression could not be evaluated against this state.

    This is *our* failure, not the user's -- a spec asking for a predicate the
    running state model does not supply, or an event missing a field a rule
    reads. It must never be turned into an alert; see `Checker._internal`.
    """


_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_BINOP: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
}


@dataclass
class Environment:
    """Every name an expression may read, assembled from three contributors.

    The checker owns `done()`, because step status is its business and no state
    tracker should know about it. The state model owns the geometry predicates.
    The event owns `src`, `dst` and `hands_in_contact`. Composing them here,
    rather than handing the evaluator one god object, is what lets a LEGO
    session run with no lattice at all: the Jenga predicates are simply absent,
    and a spec that needs them fails at setup (`missing_names`) instead of
    mid-session.
    """

    names: dict[str, Any] = field(default_factory=dict)
    funcs: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def merge(
        self,
        names: Mapping[str, Any] | None = None,
        funcs: Mapping[str, Callable[..., Any]] | None = None,
    ) -> Environment:
        return Environment(
            {**self.names, **(names or {})}, {**self.funcs, **(funcs or {})}
        )

    def bound(self) -> set[str]:
        """Names that will resolve. A `None` binding counts as unbound."""
        return {k for k, v in self.names.items() if v is not None} | set(self.funcs)


def missing_names(expr: str, available: set[str]) -> list[str]:
    """Names `expr` reads that `available` does not supply.

    Run over a whole spec before a session starts, this is what turns "a Jenga
    rule fired against a stud grid" from a mid-run crash into a refusal to
    start (IF-SPEC-1). It reports vocabulary names only; an expression that is
    not even legal is `check_expr`'s business.
    """
    if check_expr(expr):
        return []
    tree = ast.parse(expr, mode="eval")
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            used.add(node.func.id)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    known = set(STATE_VARS) | set(PREDICATES)
    return sorted(n for n in used & known if n not in available)


def evaluate(expr: str, env: Environment) -> bool:
    """Is `expr` true in `env`? Raises `EvalError`; never returns a maybe."""
    problems = check_expr(expr)
    if problems:
        # Unreachable for a validated spec. Kept because the cost of being
        # wrong about that is executing an attacker-chosen string.
        raise EvalError(f"expression is not in the permitted vocabulary: {problems[0]}")
    tree = ast.parse(expr, mode="eval")
    return bool(_visit(tree.body, env))


def _visit(node: ast.AST, env: Environment) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id in env.funcs:
            return env.funcs[node.id]
        if node.id not in env.names or env.names[node.id] is None:
            raise EvalError(f"{node.id!r} is not available in this state")
        return env.names[node.id]

    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name):
            raise EvalError("attributes are only read from state variables")
        obj = _visit(node.value, env)
        if node.attr not in ATTRS.get(node.value.id, ()):
            raise EvalError(f"{node.value.id}.{node.attr} is not in the vocabulary")
        if not hasattr(obj, node.attr):
            # A stud-grid pose has no .slot; a tower slot has no .x. Reading one
            # means the spec and the state model disagree, which is a setup bug.
            raise EvalError(
                f"{node.value.id}.{node.attr} is not carried by "
                f"{type(obj).__name__} in this state model"
            )
        return getattr(obj, node.attr)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise EvalError("only plain predicate calls are permitted")
        fn = env.funcs.get(node.func.id)
        if fn is None:
            raise EvalError(f"predicate {node.func.id!r} is not available in this state")
        return fn(*[_visit(a, env) for a in node.args])

    if isinstance(node, ast.BoolOp):
        # Short-circuit deliberately: `layer_complete(top_layer)` may be
        # expensive, and `a or layer_complete(...)` should not pay for it.
        if isinstance(node.op, ast.And):
            return all(bool(_visit(v, env)) for v in node.values)
        return any(bool(_visit(v, env)) for v in node.values)

    if isinstance(node, ast.UnaryOp):
        value = _visit(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.USub):
            return -value
        raise EvalError(f"unary {type(node.op).__name__} is not supported")

    if isinstance(node, ast.Compare):
        left = _visit(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            fn = _COMPARE.get(type(op))
            if fn is None:
                raise EvalError(f"comparison {type(op).__name__} is not supported")
            right = _visit(comparator, env)
            if not fn(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.BinOp):
        fn = _BINOP.get(type(node.op))
        if fn is None:
            raise EvalError(f"operator {type(node.op).__name__} is not supported")
        return fn(_visit(node.left, env), _visit(node.right, env))

    raise EvalError(f"{type(node).__name__} cannot be evaluated")
