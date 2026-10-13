"""Schema and lowering tests (IF-SPEC-1..4, FR-CHK-2, FR-CHK-7)."""

import pytest
from pydantic import ValidationError

from thirdeye.compiler.schema import (
    HARD_DEP_VIOLATION,
    ConstraintSpec,
    DagSpec,
    PartRef,
    Pose,
    parse_spec,
)


def test_footprint_rotation():
    part = PartRef(color="red", size="2x4")
    assert len(Pose(x=0, y=0).footprint(part)) == 8
    assert Pose(x=0, y=0, rot=0).footprint(part) != Pose(x=0, y=0, rot=90).footprint(part)
    assert max(x for x, _ in Pose(x=0, y=0, rot=90).footprint(part)) == 3


def test_size_must_be_well_formed():
    with pytest.raises(ValidationError):
        PartRef(color="red", size="big")


def test_unknown_field_is_rejected():
    # An LLM inventing a key must fail the compile, not be silently dropped.
    with pytest.raises(ValidationError):
        PartRef(color="red", size="2x4", finish="matte")


def test_duration_must_be_ordered():
    with pytest.raises(ValidationError):
        DagSpec(
            procedure="p",
            baseplate={"studs_x": 8, "studs_y": 8},
            steps=[{
                "id": "s1", "instruction": "i",
                "part": {"color": "red", "size": "2x2"}, "pose": {"x": 0, "y": 0},
                "expected_duration_s": (20, 3),
            }],
        )


def test_parse_dispatches_on_kind(dag_spec, constraint_spec):
    assert isinstance(parse_spec(dag_spec.model_dump()), DagSpec)
    assert isinstance(parse_spec(constraint_spec.model_dump()), ConstraintSpec)


def test_dag_lowers_to_constraint_form(dag_spec):
    """FR-CHK-2: the DAG is a special case of the constraint form."""
    actions = dag_spec.lower()
    assert [a.id for a in actions] == [s.id for s in dag_spec.steps]
    s2 = next(a for a in actions if a.id == "s2")
    assert {p.expr for p in s2.preconditions} == {"done('s1')"}


def test_hard_and_soft_deps_differ_only_in_consequence(dag_spec):
    """FR-CHK-7: the same predicate, routed differently."""
    s2 = next(a for a in dag_spec.lower() if a.id == "s2")
    hard = next(p for p in s2.preconditions if p.violation == HARD_DEP_VIOLATION)
    soft = next(p for p in s2.preconditions if p.violation == "out_of_order")
    assert hard.expr == soft.expr
    assert hard.on_violation == "reverify"
    assert soft.on_violation == "alert"


def test_hard_dep_never_reaches_the_user(dag_spec):
    """FR-CHK-7: no hard-dependency precondition is ever user-facing."""
    for action in dag_spec.lower():
        assert all(p.violation != HARD_DEP_VIOLATION for p in action.alerting())
        assert all(p.on_violation == "reverify" for p in action.deferring())


def test_constraint_lowering_is_identity(constraint_spec):
    actions = constraint_spec.lower()
    assert [a.id for a in actions] == [a.id for a in constraint_spec.actions]
    assert actions[0].part is None and actions[0].pose is None


def test_both_forms_produce_the_same_type(dag_spec, constraint_spec):
    """The checker sees one type, whatever the procedure."""
    assert {type(a) for a in dag_spec.lower()} == {type(a) for a in constraint_spec.lower()}
