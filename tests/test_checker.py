"""The checker's judgements (FR-CHK-1..12).

Organised by the claim each group of tests defends, not by method.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from helpers import placed

from thirdeye.checker.engine import Checker
from thirdeye.checker.rules_lego import (
    EXTRA_PART,
    REMOVED_PART,
    WRONG_BRICK,
    WRONG_POSITION,
    WRONG_ROTATION,
)
from thirdeye.compiler.schema import DagSpec, PartRef, Pose
from thirdeye.events import Event, Reverification
from thirdeye.state.lego.grid import rotations_indistinguishable

OUT_OF_ORDER = "out_of_order"
MISSED_STEP = "missed_step"


# --- a correct build is quiet (NFR-REL-2) -------------------------------------------

def test_correct_build_raises_nothing(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    for i, step in enumerate(flat_spec.steps):
        assert checker.on_event(placed(step, t=float(i))) == []
    assert checker.finish() == []
    assert checker.alerts == []
    assert all(s == "done" for s in checker.status.values())


def test_stacked_build_is_quiet_too(dag_spec: DagSpec) -> None:
    """The example spec, where every soft dependency is also a hard one."""
    checker = Checker(dag_spec)
    for i, step in enumerate(dag_spec.steps):
        assert checker.on_event(placed(step, t=float(i))) == []
    assert checker.finish() == []


def test_checklist_and_ghost_track_progress(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    assert checker.next_expected().id == "s1"
    checker.on_event(placed(flat_spec.steps[0]))
    assert checker.next_expected().id == "s2"
    assert [status for _, status, _ in checker.checklist()] == ["done", "pending", "pending"]


# --- out of order (FR-CHK-4) --------------------------------------------------------

def test_out_of_order_names_the_step_not_yet_done(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    alerts = checker.on_event(placed(flat_spec.steps[1], t=2.0))
    assert [a.code for a in alerts] == [OUT_OF_ORDER]
    assert alerts[0].rule == "s2.soft.s1"
    assert alerts[0].target == "s2"
    assert "s1" in alerts[0].text


def test_one_mistake_does_not_cascade(flat_spec: DagSpec) -> None:
    """s2 out of order must not then make s3 out of order as well.

    A flagged step was still performed, so `done('s2')` holds. Without that,
    one skipped brick would alert on every brick after it and blow the
    false-alert budget on its own.
    """
    checker = Checker(flat_spec)
    checker.on_event(placed(flat_spec.steps[1], t=1.0))
    assert checker.status["s2"] == "error"
    assert checker.on_event(placed(flat_spec.steps[2], t=2.0)) == []


def test_a_flagged_step_is_not_also_reported_missed(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    checker.on_event(placed(flat_spec.steps[1], t=1.0))
    codes = [a.code for a in checker.finish(t=9.0)]
    assert codes.count(MISSED_STEP) == 2  # s1 and s3, never performed
    assert {a.target for a in checker.alerts if a.code == MISSED_STEP} == {"s1", "s3"}


# --- missed steps (FR-CHK-3) --------------------------------------------------------

def test_session_end_reports_every_required_step_left(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    checker.on_event(placed(flat_spec.steps[0]))
    alerts = checker.finish(t=30.0)
    assert {a.target for a in alerts} == {"s2", "s3"}
    assert all(a.code == MISSED_STEP for a in alerts)
    assert "blue_2x2" in next(a.text for a in alerts if a.target == "s2")


def test_optional_steps_are_not_missed(flat_spec: DagSpec) -> None:
    flat_spec.steps[2].optional = True
    checker = Checker(flat_spec)
    for step in flat_spec.steps[:2]:
        checker.on_event(placed(step))
    assert checker.finish() == []


# --- wrong brick, position, rotation, extra, removed (FR-CHK-5, FR-CHK-6) -----------

def test_wrong_brick_at_a_wanted_cell(flat_spec: DagSpec) -> None:
    ev = Event(t=1.0, kind="ADD", part=PartRef(color="green", size="2x4"), pose=Pose(x=0, y=0))
    alerts = Checker(flat_spec).on_event(ev)
    assert [a.code for a in alerts] == [WRONG_BRICK]
    assert alerts[0].target == "s1"
    assert "red_2x4" in alerts[0].text and "green_2x4" in alerts[0].text


def test_wrong_position_measures_the_offset(flat_spec: DagSpec) -> None:
    ev = Event(t=1.0, kind="ADD", part=PartRef(color="red", size="2x4"), pose=Pose(x=1, y=0))
    alerts = Checker(flat_spec).on_event(ev)
    assert [a.code for a in alerts] == [WRONG_POSITION]
    assert alerts[0].target == "s1"
    assert "1 stud" in alerts[0].text


def test_wrong_rotation_is_distinguished_from_a_wrong_brick(flat_spec: DagSpec) -> None:
    ev = Event(t=1.0, kind="ADD", part=PartRef(color="red", size="2x4"),
               pose=Pose(x=0, y=0, rot=90))
    alerts = Checker(flat_spec).on_event(ev)
    assert [a.code for a in alerts] == [WRONG_ROTATION]
    assert alerts[0].target == "s1"


def test_extra_part(flat_spec: DagSpec) -> None:
    ev = Event(t=1.0, kind="ADD", part=PartRef(color="pink", size="1x1"), pose=Pose(x=14, y=14))
    alerts = Checker(flat_spec).on_event(ev)
    assert [a.code for a in alerts] == [EXTRA_PART]
    assert alerts[0].target == ""


def test_removing_a_done_brick_reopens_the_step(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    checker.on_event(placed(flat_spec.steps[0]))
    alerts = checker.on_event(Event(t=5.0, kind="REMOVE", pose=Pose(x=0, y=0)))
    assert [a.code for a in alerts] == [REMOVED_PART]
    assert checker.status["s1"] == "pending"
    # And it is then genuinely missing at the end.
    assert [a.code for a in checker.finish()].count(MISSED_STEP) == 3


def test_one_wrong_brick_is_one_alert(flat_spec: DagSpec) -> None:
    """The wrong brick is an attempt at s1. s2 after it is not out of order, and
    s1 is not reported again as missed: one mistake, one alert (NFR-REL-2)."""
    checker = Checker(flat_spec)
    green = Event(t=1.0, kind="ADD", part=PartRef(color="green", size="2x4"), pose=Pose(x=0, y=0))
    assert [a.code for a in checker.on_event(green)] == [WRONG_BRICK]
    assert checker.status["s1"] == "error"
    assert checker.on_event(placed(flat_spec.steps[1], t=2.0)) == []
    assert checker.on_event(placed(flat_spec.steps[2], t=3.0)) == []
    assert checker.finish() == []
    assert len(checker.alerts) == 1


def test_swapping_in_the_right_brick_completes_the_step_quietly(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    green = PartRef(color="green", size="2x4")
    checker.on_event(Event(t=1.0, kind="ADD", part=green, pose=Pose(x=0, y=0)))
    assert checker.on_event(Event(t=2.0, kind="REMOVE", part=green, pose=Pose(x=0, y=0))) == []
    assert checker.status["s1"] == "pending"
    assert checker.on_event(placed(flat_spec.steps[0], t=3.0)) == []
    assert checker.status["s1"] == "done"


def test_a_misplaced_brick_moved_to_the_right_cell_completes_the_step(
    flat_spec: DagSpec,
) -> None:
    checker = Checker(flat_spec)
    red = flat_spec.steps[0].part
    assert [a.code for a in checker.on_event(
        Event(t=1.0, kind="ADD", part=red, pose=Pose(x=1, y=0))
    )] == [WRONG_POSITION]
    # The right brick lands before the tracker drops the wrong one.
    assert checker.on_event(placed(flat_spec.steps[0], t=2.0)) == []
    assert checker.status["s1"] == "done"
    assert checker.on_event(Event(t=2.5, kind="REMOVE", part=red, pose=Pose(x=1, y=0))) == []
    assert checker.status["s1"] == "done"


def test_removing_a_brick_no_step_claims_is_silent(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    assert checker.on_event(Event(t=5.0, kind="REMOVE", pose=Pose(x=12, y=12))) == []


@pytest.mark.parametrize(
    ("size", "a", "b", "same"),
    [
        ("2x4", 0, 180, True),    # identical footprint and appearance
        ("2x4", 90, 270, True),
        ("2x4", 0, 90, False),
        ("2x2", 0, 90, True),     # square: all four look the same
        ("2x2", 0, 270, True),
        ("1x4", 0, 180, True),
    ],
)
def test_indistinguishable_rotations_never_alert(size, a, b, same) -> None:
    """We cannot see the difference, so alerting on it would be a guaranteed
    false alert (NFR-REL-2)."""
    assert rotations_indistinguishable(PartRef(color="red", size=size), a, b) is same


def test_a_brick_at_an_unobservable_rotation_is_accepted(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    turned = placed(flat_spec.steps[0], pose=Pose(x=0, y=0, rot=180))
    assert checker.on_event(turned) == []
    assert checker.status["s1"] == "done"


# --- hard dependencies never reach the user (FR-CHK-7, FR-CHK-8) --------------------

class _RecordingState:
    """Records re-verification requests; supplies no predicates."""

    def __init__(self) -> None:
        self.requested: list[set[str]] = []

    def request_reverify(self, step_ids: set[str]) -> None:
        self.requested.append(set(step_ids))


def test_unmet_hard_dependency_defers_instead_of_alerting(dag_spec: DagSpec) -> None:
    state = _RecordingState()
    checker = Checker(dag_spec, state=state)
    # s2 sits on s1. Seeing s2 placed first means we misread the plate.
    assert checker.on_event(placed(dag_spec.steps[1], t=3.0)) == []
    assert checker.alerts == []
    assert state.requested == [{"s1"}]
    assert checker.status["s2"] == "pending"


def test_a_confirmed_perception_miss_costs_the_user_nothing(dag_spec: DagSpec) -> None:
    checker = Checker(dag_spec, state=_RecordingState())
    checker.on_event(placed(dag_spec.steps[1], t=3.0))
    alerts = checker.on_reverify(Reverification(t=4.0, supported={"s1"}))
    assert alerts == []
    assert checker.perception_misses == ["s1"]
    assert checker.status == {"s1": "done", "s2": "done"}
    assert checker.finish() == []


def test_a_dependency_committed_before_the_re_read_is_not_our_miss(
    dag_spec: DagSpec,
) -> None:
    """The tracker caught up on its own; charging our recall for that would be
    as wrong as ignoring a real miss."""
    checker = Checker(dag_spec, state=_RecordingState())
    checker.on_event(placed(dag_spec.steps[1], t=3.0))
    checker.on_event(placed(dag_spec.steps[0], t=3.5))
    alerts = checker.on_reverify(Reverification(t=4.0, supported={"s1"}))
    assert alerts == []
    assert checker.perception_misses == []
    assert checker.status == {"s1": "done", "s2": "done"}


def test_the_same_brick_seen_again_is_not_an_extra_part(dag_spec: DagSpec) -> None:
    """A re-read credits s1 before the tracker commits it; the commit that
    follows is the same brick, not a second one."""
    checker = Checker(dag_spec, state=_RecordingState())
    checker.on_event(placed(dag_spec.steps[1], t=3.0))
    checker.on_reverify(Reverification(t=4.0, supported={"s1"}))
    assert checker.on_event(placed(dag_spec.steps[0], t=4.5)) == []
    assert checker.alerts == []


def test_a_genuine_skip_is_reported_only_after_the_re_read(dag_spec: DagSpec) -> None:
    checker = Checker(dag_spec, state=_RecordingState())
    checker.on_event(placed(dag_spec.steps[1], t=3.0))
    alerts = checker.on_reverify(Reverification(t=4.0, supported=set()))
    codes = [a.code for a in alerts]
    assert MISSED_STEP in codes and OUT_OF_ORDER in codes
    assert checker.perception_misses == []
    # s2 was observed, so it completes; s1 is not reported a second time.
    assert checker.status["s2"] == "error"
    assert checker.finish() == []


def test_an_unanswered_re_read_does_not_become_a_missed_step(dag_spec: DagSpec) -> None:
    """If the view never clears we own that, and the brick we did see counts."""
    checker = Checker(dag_spec, state=_RecordingState())
    checker.on_event(placed(dag_spec.steps[1], t=3.0))
    alerts = checker.finish(t=30.0)
    assert {a.target for a in alerts} == {"s1"}
    assert checker.status["s2"] in ("done", "error")


# --- soft warnings (FR-CHK-9) -------------------------------------------------------

def test_low_confidence_downgrades_to_yellow(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec, confidence_threshold=0.8)
    alerts = checker.on_event(placed(flat_spec.steps[1], t=1.0, confidence=0.4))
    assert [a.severity for a in alerts] == ["warning"]


def test_confident_events_stay_red(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec, confidence_threshold=0.8)
    alerts = checker.on_event(placed(flat_spec.steps[1], t=1.0, confidence=0.95))
    assert [a.severity for a in alerts] == ["alert"]


def test_session_end_misses_are_never_downgraded(flat_spec: DagSpec) -> None:
    """A step that is simply absent is not a confidence question."""
    checker = Checker(flat_spec, confidence_threshold=0.99)
    assert all(a.severity == "alert" for a in checker.finish())


# --- attribution and logging (FR-CHK-11, FR-CHK-12) --------------------------------

def test_every_alert_names_a_rule(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    checker.on_event(Event(t=1.0, kind="ADD", part=PartRef(color="green", size="2x4"),
                           pose=Pose(x=0, y=0)))
    checker.on_event(placed(flat_spec.steps[2], t=2.0))
    checker.finish(t=3.0)
    assert checker.alerts
    assert all(a.rule and a.code and a.text for a in checker.alerts)


def test_the_log_records_events_alerts_and_statuses(flat_spec: DagSpec) -> None:
    checker = Checker(flat_spec)
    checker.on_event(placed(flat_spec.steps[1], t=1.0))
    kinds = {e.kind for e in checker.log}
    assert {"event", "alert", "status"} <= kinds
    assert all(e.t >= 0 for e in checker.log)


def test_a_temporal_hint_can_only_make_a_step_active(flat_spec: DagSpec) -> None:
    """FR-CHK-11: no alert may rest on a learned model's output alone."""
    checker = Checker(flat_spec)
    hint = Event(t=1.0, kind="STEP_START", source="temporal", part=flat_spec.steps[0].part)
    assert checker.on_event(hint) == []
    assert checker.status["s1"] == "active"
    # And the state diff still completes it.
    assert checker.on_event(placed(flat_spec.steps[0], t=2.0)) == []
    assert checker.status["s1"] == "done"


# --- one engine for both procedure forms (FR-CHK-2) --------------------------------

def test_the_engine_never_branches_on_procedure_type() -> None:
    """The generality claim is structural, so assert it structurally.

    If this fails, `lower()` has stopped carrying its weight and the tier-3
    result in plan 1.2 is no longer supported by the code.
    """
    source = Path("thirdeye/checker/engine.py").read_text()
    tree = ast.parse(source)
    spec_types = {"DagSpec", "ConstraintSpec", "Step", "Action"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "isinstance":
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not (names & spec_types), f"engine branches on {names & spec_types}"
        if isinstance(node, ast.Attribute) and node.attr == "kind":
            assert getattr(node.value, "id", "") != "spec", "engine reads spec.kind"
