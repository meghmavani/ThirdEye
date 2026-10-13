"""The evaluator never executes Python, and never guesses (FR-CMP-5)."""

import pytest

from thirdeye.checker.evaluate import Environment, EvalError, evaluate, missing_names
from thirdeye.events import Slot


@pytest.fixture
def env() -> Environment:
    return Environment(
        names={"top_layer": 17, "top_complete_layer": 16, "collapse_detected": False,
               "hands_in_contact": 1, "src": Slot(5, 1), "dst": Slot(17, 2)},
        funcs={"done": lambda s: s == "s1",
               "layer_complete": lambda n: n == 16,
               "legal_top_slots": lambda: {Slot(17, 2)},
               "occupancy": lambda layer, slot: 1},
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("src.layer <= top_complete_layer - 1", True),
        ("src.layer <= 3", False),
        ("hands_in_contact == 1", True),
        ("not collapse_detected", True),
        ("dst.layer == top_layer or layer_complete(top_layer)", True),
        ("dst.layer == 3 or layer_complete(top_layer)", False),
        ("dst in legal_top_slots()", True),
        ("src in legal_top_slots()", False),
        ("done('s1')", True),
        ("done('s2')", False),
        ("occupancy(5, 1) == 1 and not done('s2')", True),
        ("1 <= src.layer <= 10", True),  # chained comparison
        ("-src.layer < 0", True),
    ],
)
def test_evaluates_the_vocabulary(expr: str, expected: bool, env: Environment) -> None:
    assert evaluate(expr, env) is expected


def test_unbound_name_is_our_failure_not_a_verdict(env: Environment) -> None:
    """A missing binding raises rather than returning False -- returning False
    would be an alert against the user for a gap in our own state."""
    bare = Environment(funcs={"done": lambda s: False})
    with pytest.raises(EvalError, match="top_layer"):
        evaluate("top_layer == 3", bare)


def test_missing_attribute_names_the_state_model(env: Environment) -> None:
    """A stud-grid pose has no .slot; the error must say so, not crash opaquely."""
    from thirdeye.compiler.schema import Pose

    grid = env.merge({"src": Pose(x=1, y=2)})
    with pytest.raises(EvalError, match="not carried by"):
        evaluate("src.slot == 0", grid)


def test_none_binding_counts_as_unbound(env: Environment) -> None:
    """An event that did not carry `src` must not silently read as falsy."""
    without = env.merge({"src": None})
    with pytest.raises(EvalError, match="not available"):
        evaluate("src.layer <= 3", without)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "[x for x in ()]",
        "src.__class__",
        "done('s1') if True else False",
        "lambda: 1",
    ],
)
def test_refuses_anything_outside_the_vocabulary(expr: str, env: Environment) -> None:
    with pytest.raises(EvalError, match="permitted vocabulary"):
        evaluate(expr, env)


def test_missing_names_reports_the_gap() -> None:
    assert missing_names("src.layer <= top_complete_layer - 1", {"src"}) == [
        "top_complete_layer"
    ]
    assert missing_names("done('s1')", {"done"}) == []
    assert missing_names("dst in legal_top_slots()", {"dst"}) == ["legal_top_slots"]


def test_boolean_operators_short_circuit(env: Environment) -> None:
    """`or` must not pay for a predicate it does not need -- NFR-PERF-8 is 1 ms."""
    calls = []
    counting = env.merge(funcs={"layer_complete": lambda n: calls.append(n) or True})
    assert evaluate("hands_in_contact == 1 or layer_complete(top_layer)", counting) is True
    assert calls == []
