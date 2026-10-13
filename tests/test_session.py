"""The tracker-to-checker join (plan 4.5, FR-CHK-7 with FR-STA-6)."""

from __future__ import annotations

from thirdeye.compiler.schema import ConstraintSpec, DagSpec, PartRef, Pose
from thirdeye.events import Event, Slot
from thirdeye.session import Session, state_for
from thirdeye.state.jenga.lattice import TowerLattice, full_tower
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import Reading


def see(session: Session, t: float, steps, hands: bool = False) -> list:
    """Show the tracker the bricks these steps place, then judge."""
    readings = [Reading(part=s.part, pose=s.pose, residual=0.0) for s in steps]
    session.state.observe(t, readings, hands_present=hands)
    return session.pump()


def _pulled(layer: int, slot: int):
    rows = [list(r) for r in full_tower()]
    rows[layer][slot] = 0
    return tuple(tuple(r) for r in rows)


def test_a_session_builds_the_state_model_its_spec_declares(
    dag_spec: DagSpec, constraint_spec: ConstraintSpec
) -> None:
    assert isinstance(state_for(dag_spec), BuildState)
    assert isinstance(state_for(constraint_spec), TowerLattice)
    assert isinstance(Session(dag_spec).state, BuildState)


def test_a_correct_build_runs_through_quietly(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    placed = []
    for i, step in enumerate(flat_spec.steps):
        placed.append(step)
        assert see(session, i * 2.0, placed) == []
        assert see(session, i * 2.0 + 1.0, placed) == []
    assert session.finish() == []
    assert session.alerts == []
    assert "3/3 steps done" in session.summary()


def test_a_skipped_step_is_caught_at_the_end(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    for t in (0.0, 1.0):
        see(session, t, [flat_spec.steps[0], flat_spec.steps[2]])
    alerts = session.finish(t=30.0)
    assert [a.target for a in alerts] == ["s2"]
    assert [a.code for a in alerts] == ["missed_step"]


def test_out_of_order_fires_through_the_tracker(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    for t in (0.0, 1.0):
        alerts = see(session, t, [flat_spec.steps[1]])
    assert [a.code for a in alerts] == ["out_of_order"]


def test_bricks_settling_together_are_judged_bottom_up(dag_spec: DagSpec) -> None:
    """s2 sits on s1 and both settle in the same frame. Listing the top brick
    first must not make the checker doubt the one holding it up."""
    session = Session(dag_spec)
    s1, s2 = dag_spec.steps
    for t in (0.0, 1.0):
        assert see(session, t, [s2, s1]) == []
    assert session.state.pending_reverify() == set()
    assert session.checker.status == {"s1": "done", "s2": "done"}
    assert session.finish() == []


def test_a_perception_miss_is_resolved_by_the_next_clear_reading(dag_spec: DagSpec) -> None:
    """s1 is on the plate, but came into view too late to commit before s2.

    The re-read is a later frame, not the one that raised the doubt; it finds
    s1, the miss is ours, and the user hears nothing (FR-CHK-7, FR-CHK-8).
    """
    session = Session(dag_spec)
    s1, s2 = dag_spec.steps
    assert see(session, 0.0, [s2]) == []
    assert see(session, 0.6, [s2, s1]) == []          # s2 commits; deferred
    assert session.state.pending_reverify() == {"s1"}
    assert see(session, 0.7, [s2, s1]) == []          # the re-read
    assert session.checker.perception_misses == ["s1"]
    assert see(session, 1.2, [s2, s1]) == []          # s1's own late commit
    assert session.finish() == []
    assert session.alerts == []


def test_a_genuine_skip_under_a_brick_is_reported_not_written_off(dag_spec: DagSpec) -> None:
    """s2 is stacked on s1, and s1 really is absent.

    The re-read must not flatter our own recall by calling this a perception
    miss -- `_observed` asks the tracker, and the tracker does not see s1.
    """
    session = Session(dag_spec)
    for t in (0.0, 1.0):
        assert see(session, t, [dag_spec.steps[1]]) == []   # deferred, not judged
    alerts = see(session, 1.1, [dag_spec.steps[1]])
    assert {a.code for a in alerts} == {"missed_step", "out_of_order"}
    assert session.checker.perception_misses == []
    assert [a.code for a in session.finish()] == []     # reported once, not twice


def test_a_re_read_waits_for_the_hand_to_leave(dag_spec: DagSpec) -> None:
    session = Session(dag_spec)
    for t in (0.0, 1.0):
        see(session, t, [dag_spec.steps[1]])
    assert see(session, 1.1, [], hands=True) == []
    assert session.state.pending_reverify() == {"s1"}
    assert {a.code for a in see(session, 1.2, [dag_spec.steps[1]])} == {
        "missed_step",
        "out_of_order",
    }


def test_nothing_is_judged_while_a_hand_covers_the_plate(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    for t in (0.0, 0.5, 1.0):
        assert see(session, t, [flat_spec.steps[1]], hands=True) == []
    assert session.alerts == []


def test_a_temporal_hint_is_submitted_alongside_the_tracker(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    hint = Event(t=0.1, kind="STEP_START", source="temporal", part=flat_spec.steps[0].part)
    assert session.submit(hint) == []
    assert session.checklist()[0][1] == "active"


def test_a_wrong_brick_reaches_the_user_through_the_join(flat_spec: DagSpec) -> None:
    session = Session(flat_spec)
    stray = Reading(part=PartRef(color="green", size="2x4"), pose=Pose(x=0, y=0), residual=0.0)
    for t in (0.0, 1.0):
        session.state.observe(t, [stray])
        alerts = session.pump()
    assert [a.code for a in alerts] == ["wrong_brick"]
    assert "1 alerts" in session.summary()


# --- the Jenga path through the same join ------------------------------------------

def test_bare_halves_of_a_move_are_not_blamed_on_the_player(
    constraint_spec: ConstraintSpec,
) -> None:
    session = Session(constraint_spec)
    assert isinstance(session.state, TowerLattice)
    session.state.observe(_pulled(17, 0), t=5.0)
    assert session.pump() == []


def test_the_tracker_s_halves_are_paired_into_a_judged_move(
    constraint_spec: ConstraintSpec,
) -> None:
    """Pull from layer 17 -- the top complete layer -- and put it on top: the
    lattice reports two slot changes, the session judges one move."""
    session = Session(constraint_spec)
    session.state.observe(_pulled(17, 0), t=5.0, hands_in_contact=1)
    assert session.pump() == []
    rows = [list(r) for r in _pulled(17, 0)] + [[1, 0, 0]]
    session.state.observe(rows, t=8.0)
    assert [a.code for a in session.pump()] == ["illegal_source_layer"]


def test_a_two_handed_pull_is_caught_through_the_tracker(
    constraint_spec: ConstraintSpec,
) -> None:
    session = Session(constraint_spec)
    session.state.observe(_pulled(3, 1), t=5.0, hands_in_contact=2)
    session.pump()
    rows = [list(r) for r in _pulled(3, 1)] + [[1, 0, 0]]
    session.state.observe(rows, t=8.0)
    assert [a.code for a in session.pump()] == ["two_handed_move"]


def test_an_unseen_hand_count_switches_off_only_that_rule(
    constraint_spec: ConstraintSpec,
) -> None:
    session = Session(constraint_spec)
    session.state.observe(_pulled(3, 1), t=5.0)
    session.pump()
    rows = [list(r) for r in _pulled(3, 1)] + [[1, 0, 0]]
    session.state.observe(rows, t=8.0)
    assert session.pump() == []
    assert "c2" in session.checker.unevaluable


def test_a_paired_move_is_judged(constraint_spec: ConstraintSpec) -> None:
    session = Session(constraint_spec)
    alerts = session.submit(
        Event(t=5.0, kind="MOVE", src=Slot(17, 0), dst=Slot(18, 0), hands_in_contact=1)
    )
    assert [a.code for a in alerts] == ["illegal_source_layer"]


def test_a_collapse_ends_a_jenga_session(constraint_spec: ConstraintSpec) -> None:
    session = Session(constraint_spec)
    session.submit(Event(t=9.0, kind="COLLAPSE"))
    assert session.outcome == "game_over"
    assert "ended: game_over" in session.summary()
