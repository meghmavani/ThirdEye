"""The tower lattice and the constraint path (FR-STA-7, FR-STA-8, FR-CHK-10).

These tests carry the tier-3 claim of plan 1.2: the *spec format* is general, not
just the weights. Every Jenga rule here is enforced from `specs/example_constraints.json`
through the same engine that runs LEGO -- no rule is implemented in Python.
"""

from __future__ import annotations

import pytest

from thirdeye.checker.engine import Checker, SpecNotRunnable
from thirdeye.checker.rules_jenga import MovePairer
from thirdeye.compiler.schema import ConstraintSpec
from thirdeye.events import Event, Slot
from thirdeye.state.jenga.lattice import (
    TowerLattice,
    empty_tower,
    full_tower,
    layer_complete,
    legal_top_slots,
    orientation,
    top_complete_layer,
    top_layer,
)


def tower(rows: dict[int, tuple[int, int, int]] | None = None, layers: int = 18):
    """A full tower with the named layers overridden."""
    snap = [list(row) for row in full_tower(layers)]
    for layer, slots in (rows or {}).items():
        while len(snap) <= layer:
            snap.append([0, 0, 0])
        snap[layer] = list(slots)
    return tuple(tuple(r) for r in snap)


def move(src: Slot, dst: Slot, t: float = 1.0, hands: int = 1) -> Event:
    return Event(t=t, kind="MOVE", src=src, dst=dst, hands_in_contact=hands)


# --- the lattice itself (FR-STA-7) --------------------------------------------------

def test_a_fresh_tower_reads_as_eighteen_complete_layers() -> None:
    snap = full_tower()
    assert top_layer(snap) == 17
    assert top_complete_layer(snap) == 17
    assert legal_top_slots(snap) == {Slot(18, s) for s in range(3)}


def test_an_empty_table_has_no_top_layer() -> None:
    snap = empty_tower()
    assert top_layer(snap) == -1
    assert top_complete_layer(snap) == -1
    assert legal_top_slots(snap) == {Slot(0, s) for s in range(3)}


def test_an_unfinished_top_layer_offers_only_its_gaps() -> None:
    snap = tower({18: (1, 0, 0)})
    assert top_layer(snap) == 18
    assert top_complete_layer(snap) == 17
    assert legal_top_slots(snap) == {Slot(18, 1), Slot(18, 2)}


def test_a_gap_lower_down_does_not_change_the_top_complete_layer() -> None:
    """Removing from layer 5 must not make layer 4 the top complete layer --
    every rule about legal sources reads that number."""
    assert top_complete_layer(tower({5: (1, 0, 1)})) == 17


def test_layers_alternate_orientation() -> None:
    assert [orientation(n) for n in range(4)] == [0, 90, 0, 90]


def test_layer_complete_is_false_outside_the_tower() -> None:
    snap = full_tower()
    assert layer_complete(snap, 17) is True
    assert layer_complete(snap, 18) is False
    assert layer_complete(snap, -1) is False


# --- readings become events (FR-STA-8) ---------------------------------------------

def test_a_reading_diffs_into_a_removal_then_an_arrival() -> None:
    lattice = TowerLattice()
    lattice.observe(tower({5: (1, 0, 1), 18: (1, 0, 0)}), t=4.0)
    events = lattice.diff()
    assert [e.kind for e in events] == ["REMOVE", "ADD"]
    assert events[0].src == Slot(5, 1)
    assert events[1].dst == Slot(18, 0)
    assert all(e.t == 4.0 for e in events)


def test_diff_is_consumed() -> None:
    lattice = TowerLattice()
    lattice.observe(tower({5: (1, 0, 1)}))
    assert lattice.diff()
    assert lattice.diff() == []


def test_a_collapse_is_reported_as_a_terminal_event() -> None:
    lattice = TowerLattice()
    lattice.collapse_detected = True
    lattice.observe(empty_tower(), t=9.0)
    assert [e.kind for e in lattice.diff()][-1] == "COLLAPSE"


def test_a_reading_with_the_wrong_slot_count_is_refused() -> None:
    with pytest.raises(ValueError, match="3 slots"):
        TowerLattice().observe([[1, 1]])


# --- the rulebook is enforced from the spec (FR-CHK-10) ----------------------------

def test_a_rulebook_cannot_run_without_a_tower(constraint_spec: ConstraintSpec) -> None:
    """A Jenga spec judged against no state must refuse to start, not fail
    silently halfway through a demo (IF-SPEC-1)."""
    with pytest.raises(SpecNotRunnable, match="top_complete_layer"):
        Checker(constraint_spec)


def test_a_legal_move_is_quiet(constraint_spec: ConstraintSpec) -> None:
    lattice = TowerLattice()
    checker = Checker(constraint_spec, state=lattice)
    assert checker.on_event(move(Slot(5, 1), Slot(18, 0))) == []
    assert checker.alerts == []


def test_taking_from_the_top_complete_layer_is_illegal(constraint_spec) -> None:
    checker = Checker(constraint_spec, state=TowerLattice())
    alerts = checker.on_event(move(Slot(17, 0), Slot(18, 0)))
    assert [a.code for a in alerts] == ["illegal_source_layer"]
    assert alerts[0].rule == "c1"
    assert "L17" in alerts[0].text and "L18s0" in alerts[0].text


def test_a_two_handed_pull_is_illegal(constraint_spec) -> None:
    checker = Checker(constraint_spec, state=TowerLattice())
    alerts = checker.on_event(move(Slot(5, 1), Slot(18, 0), hands=2))
    assert [a.code for a in alerts] == ["two_handed_move"]
    assert alerts[0].rule == "c2"


def test_returning_a_block_to_a_gap_lower_down_is_illegal(constraint_spec) -> None:
    lattice = TowerLattice(initial=tower({5: (1, 0, 1)}))
    checker = Checker(constraint_spec, state=lattice)
    alerts = checker.on_event(move(Slot(3, 0), Slot(5, 1)))
    assert [a.code for a in alerts] == ["wrong_replacement"]
    assert alerts[0].rule == "c4"


def test_starting_a_layer_early_is_illegal(constraint_spec) -> None:
    lattice = TowerLattice(initial=tower({18: (1, 0, 0)}))
    checker = Checker(constraint_spec, state=lattice)
    codes = {a.code for a in checker.on_event(move(Slot(5, 1), Slot(19, 0)))}
    assert "layer_started_early" in codes


def test_rules_are_judged_against_the_state_before_the_move() -> None:
    """The reason `predicates()` binds the previous reading.

    The tracker updates only after the hands leave, so the illegal move has
    already happened by the time we judge it. Placing the first block of layer
    19 makes layer 19 the top layer, which satisfies "finish a layer before
    starting the next" -- read post-move, the violation erases its own evidence.
    """
    pre = tower({18: (1, 0, 0)})
    post = tower({5: (1, 0, 1), 18: (1, 0, 0), 19: (1, 0, 0)})
    lattice = TowerLattice(initial=pre)
    lattice.observe(post, t=2.0)

    assert lattice.before() == pre
    # The trap: read after the move, `dst.layer == top_layer` is satisfied by
    # the very block that broke the rule, and c3 passes.
    assert top_layer(lattice.current()) == 19
    assert top_layer(lattice.before()) == 18
    assert Slot(19, 0) not in legal_top_slots(lattice.before())

    names, funcs = lattice.predicates()
    assert names["top_layer"] == 18
    assert funcs["layer_complete"](18) is False


def test_a_move_across_two_readings_is_judged_against_the_tower_before_it() -> None:
    """Pull in one reading, place in the next. The reading between shows a gap
    in layer 17 with nothing on top; judging the placement against that would
    make one pull from the top complete layer look like three fouls."""
    pre = full_tower()
    lattice = TowerLattice(initial=pre)
    lattice.observe(tower({17: (0, 1, 1)}), t=2.0, hands_in_contact=1)
    lattice.observe(tower({17: (0, 1, 1), 18: (1, 0, 0)}), t=6.0)
    assert lattice.before() == pre

    # The next move starts from the tower at rest, block back on top.
    lattice.observe(tower({5: (1, 0, 1), 17: (0, 1, 1), 18: (1, 0, 0)}), t=9.0)
    assert lattice.before() == tower({17: (0, 1, 1), 18: (1, 0, 0)})


def test_a_pull_carries_the_hand_count_that_watched_it() -> None:
    lattice = TowerLattice()
    lattice.observe(tower({4: (1, 0, 1)}), t=2.0, hands_in_contact=2)
    [removal] = lattice.diff()
    assert removal.hands_in_contact == 2


def test_an_unknown_hand_count_silences_only_its_own_rule(constraint_spec) -> None:
    """One missing perception signal must not switch off the other three rules."""
    checker = Checker(constraint_spec, state=TowerLattice())
    blind = Event(t=1.0, kind="MOVE", src=Slot(17, 0), dst=Slot(18, 0))
    alerts = checker.on_event(blind)
    assert [a.code for a in alerts] == ["illegal_source_layer"]
    assert checker.unevaluable == ["c2"]


def test_a_collapse_ends_the_session(constraint_spec) -> None:
    checker = Checker(constraint_spec, state=TowerLattice())
    alerts = checker.on_event(Event(t=12.0, kind="COLLAPSE"))
    assert checker.outcome == "game_over"
    assert [a.code for a in alerts] == ["game_over"]
    # Nothing is judged after the tower is on the table.
    assert checker.on_event(move(Slot(17, 0), Slot(18, 0), t=13.0)) == []


def test_the_terminal_expression_also_fires_from_state(constraint_spec) -> None:
    lattice = TowerLattice()
    lattice.collapse_detected = True
    checker = Checker(constraint_spec, state=lattice)
    checker.on_event(move(Slot(5, 1), Slot(18, 0)))
    assert checker.outcome == "game_over"


def test_a_rulebook_has_no_steps_to_miss(constraint_spec) -> None:
    """Constraint actions are repeatable, so the session-end sweep is silent."""
    checker = Checker(constraint_spec, state=TowerLattice())
    checker.on_event(move(Slot(5, 1), Slot(18, 0)))
    assert checker.finish() == []
    assert checker.status == {"move": "pending"}


# --- pairing the halves of a move --------------------------------------------------

def test_a_removal_and_the_arrival_that_follows_make_one_move() -> None:
    pairer = MovePairer()
    removal = Event(t=1.0, kind="REMOVE", src=Slot(5, 1), hands_in_contact=2, confidence=0.9)
    assert pairer.feed(removal) == []
    out = pairer.feed(Event(t=3.0, kind="ADD", dst=Slot(18, 0), confidence=0.7))
    assert len(out) == 1
    paired = out[0]
    assert paired.kind == "MOVE"
    assert (paired.src, paired.dst) == (Slot(5, 1), Slot(18, 0))
    assert paired.hands_in_contact == 2  # the rule is about the pull
    assert paired.confidence == pytest.approx(0.7)  # the weaker half governs
    assert paired.t == 3.0


def test_a_removal_never_completed_expires_rather_than_alerting() -> None:
    pairer = MovePairer(window_s=5.0)
    pairer.feed(Event(t=1.0, kind="REMOVE", src=Slot(5, 1)))
    assert pairer.feed(Event(t=20.0, kind="ADD", dst=Slot(18, 0))) == [
        Event(t=20.0, kind="ADD", dst=Slot(18, 0))
    ]
    assert [e.src for e in pairer.dropped] == [Slot(5, 1)]


def test_a_collapse_passes_straight_through() -> None:
    pairer = MovePairer()
    collapse = Event(t=9.0, kind="COLLAPSE")
    assert pairer.feed(collapse) == [collapse]


def test_a_reading_flows_from_lattice_through_pairer_to_checker(constraint_spec) -> None:
    """The whole online Jenga path, minus the cameras."""
    lattice = TowerLattice()
    checker = Checker(constraint_spec, state=lattice)
    pairer = MovePairer()

    lattice.observe(tower({17: (0, 1, 1), 18: (1, 0, 0)}), t=6.0)
    alerts = []
    for ev in lattice.diff():
        for paired in pairer.feed(ev):
            alerts += checker.on_event(
                Event(**{**paired.__dict__, "hands_in_contact": 1})
            )
    assert [a.code for a in alerts] == ["illegal_source_layer"]
