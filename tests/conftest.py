"""Fixtures built from the spec examples in the plan (sections 4.2 and 4.3)."""

from pathlib import Path

import pytest

from thirdeye.compiler.schema import (
    Baseplate,
    ConstraintSpec,
    DagSpec,
    PartRef,
    Pose,
    Step,
    load_spec,
)

SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.fixture
def dag_spec() -> DagSpec:
    spec = load_spec(SPECS / "example_dag.json")
    assert isinstance(spec, DagSpec)
    return spec


@pytest.fixture
def constraint_spec() -> ConstraintSpec:
    spec = load_spec(SPECS / "example_constraints.json")
    assert isinstance(spec, ConstraintSpec)
    return spec


@pytest.fixture
def flat_spec() -> DagSpec:
    """Three bricks side by side on one layer, ordered by the manual only.

    Deliberately has no *hard* dependencies: with nothing stacked, nothing
    supports anything, so a violation of the stated order is unambiguously an
    out-of-order error rather than a deferred re-verification. The example spec
    cannot show that, because there every soft dependency is also a hard one.
    """
    return DagSpec(
        procedure="flat_row",
        baseplate=Baseplate(studs_x=16, studs_y=16),
        steps=[
            Step(
                id="s1",
                instruction="Place the red 2x4 at the left edge",
                part=PartRef(color="red", size="2x4"),
                pose=Pose(x=0, y=0),
            ),
            Step(
                id="s2",
                instruction="Place the blue 2x2 beside it",
                part=PartRef(color="blue", size="2x2"),
                pose=Pose(x=4, y=0),
                soft_depends_on=["s1"],
            ),
            Step(
                id="s3",
                instruction="Place the yellow 1x4 beside that",
                part=PartRef(color="yellow", size="1x4"),
                pose=Pose(x=8, y=0),
                soft_depends_on=["s2"],
            ),
        ],
    )
