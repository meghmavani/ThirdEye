"""The jigsaw spec form, tracker and checker policy (FR-JIG-1, FR-STA-10, FR-CHK-3..6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from thirdeye.compiler.jigsaw import border_first
from thirdeye.compiler.schema import CellPose, JigsawSpec, PieceRef, load_spec, parse_spec
from thirdeye.compiler.validate import errors, validate
from thirdeye.session import Session
from thirdeye.state.jigsaw.piece_grid import PieceGridState, PieceReading

SPECS = Path(__file__).resolve().parents[1] / "specs"


def spec_4x3() -> JigsawSpec:
    spec = load_spec(SPECS / "jigsaw_forest_12.json")
    assert isinstance(spec, JigsawSpec)
    return spec


def reading(piece: str | None, col: int, row: int, rot: int = 0, conf: float = 1.0):
    return PieceReading(col=col, row=row, piece=piece, rot=rot, confidence=conf)


def place(session: Session, t: float, readings: list[PieceReading]) -> list:
    """Show the board for one second so the stability filter commits it."""
    out = []
    state = session.state
    assert isinstance(state, PieceGridState)
    for k in range(12):
        state.observe(t + k * 0.1, readings)
        out += session.pump()
    return out


# --- the spec ------------------------------------------------------------------------

def test_the_forest_spec_has_the_expected_shape() -> None:
    spec = spec_4x3()
    assert (spec.grid.cols, spec.grid.rows, len(spec.steps)) == (4, 3, 12)
    corners, edges, interior = spec.steps[:4], spec.steps[4:10], spec.steps[10:]
    assert all(s.soft_depends_on == [] for s in corners)
    assert all(set(s.soft_depends_on) == {c.id for c in corners} for s in edges)
    assert all(set(s.soft_depends_on) == {e.id for e in edges} for s in interior)
    assert all(s.hard_depends_on == [] for s in spec.steps)
    assert not errors(validate(spec))


def test_a_jigsaw_spec_round_trips_through_the_discriminated_parser() -> None:
    spec = spec_4x3()
    assert parse_spec(spec.model_dump(mode="json")) == spec


def test_the_stud_grid_form_still_parses_as_a_lego_spec() -> None:
    lego = load_spec(SPECS / "model_a.json")
    assert lego.state_model == "stud_grid"


def test_a_piece_pose_rejects_a_rotation_that_is_not_a_quarter_turn() -> None:
    with pytest.raises(ValidationError):
        CellPose(col=0, row=0, rot=45)  # type: ignore[arg-type]


def test_the_validator_catches_a_cell_off_the_grid_and_a_double_booking() -> None:
    spec = border_first("t", 2, 2)
    off = spec.model_copy(deep=True)
    off.steps[0].pose = CellPose(col=5, row=0)
    assert any(p.code == "out_of_bounds" for p in errors(validate(off)))

    double = spec.model_copy(deep=True)
    double.steps[1].pose = double.steps[0].pose
    assert {p.code for p in errors(validate(double))} >= {"collision"}

    twice = spec.model_copy(deep=True)
    twice.steps[1].part = PieceRef(piece=twice.steps[0].part.piece)
    assert any(p.code == "duplicate_piece" for p in errors(validate(twice)))


def test_the_validator_rejects_a_dependency_cycle() -> None:
    spec = border_first("t", 3, 3)
    spec.steps[0].soft_depends_on = [spec.steps[-1].id]
    assert any(p.code == "cycle" for p in errors(validate(spec)))


def test_a_hard_dependency_on_a_flat_jigsaw_is_flagged() -> None:
    spec = border_first("t", 2, 2)
    spec.steps[1].hard_depends_on = [spec.steps[0].id]
    assert any(p.code == "unexpected_hard_dependency" for p in validate(spec))


# --- the tracker ---------------------------------------------------------------------

def test_a_piece_is_placed_only_after_it_holds_still() -> None:
    state = PieceGridState(cols=4, rows=3)
    state.observe(0.0, [reading("r0c0", 0, 0)])
    state.observe(0.3, [reading("r0c0", 0, 0)])
    assert state.diff() == []
    state.observe(0.6, [reading("r0c0", 0, 0)])
    [ev] = state.diff()
    assert ev.kind == "ADD" and ev.part == PieceRef(piece="r0c0")
    assert ev.pose == CellPose(col=0, row=0, rot=0)


def test_nothing_is_committed_while_a_hand_is_over_the_board() -> None:
    state = PieceGridState(cols=4, rows=3)
    for k in range(10):
        state.observe(k * 0.1, [reading("r0c0", 0, 0)], hands_present=True)
    assert state.diff() == []
    state.observe(1.0, [reading("r0c0", 0, 0)])
    state.observe(1.6, [reading("r0c0", 0, 0)])
    assert len(state.diff()) == 1


def test_an_unidentified_piece_occupies_its_cell_but_raises_no_event() -> None:
    state = PieceGridState(cols=4, rows=3)
    for k in range(20):
        state.observe(k * 0.1, [reading(None, 1, 1)])
    assert state.diff() == []
    assert state.current() == {(1, 1): None}


def test_a_placed_piece_that_turns_unidentified_is_held_not_removed() -> None:
    state = PieceGridState(cols=4, rows=3)
    for k in range(8):
        state.observe(k * 0.1, [reading("r0c0", 0, 0)])
    state.diff()
    for k in range(8, 30):
        state.observe(k * 0.1, [reading(None, 0, 0)])
    assert state.diff() == []
    assert (0, 0) in state.placed


def test_a_turned_piece_is_reported_as_a_removal_then_a_placement() -> None:
    state = PieceGridState(cols=4, rows=3)
    for k in range(8):
        state.observe(k * 0.1, [reading("r0c0", 0, 0, rot=0)])
    state.diff()
    for k in range(8, 20):
        state.observe(k * 0.1, [reading("r0c0", 0, 0, rot=90)])
    assert [e.kind for e in state.diff()] == ["REMOVE", "ADD"]


def test_a_reading_off_the_board_is_dropped_and_counted() -> None:
    state = PieceGridState(cols=2, rows=2)
    state.observe(0.0, [reading("r0c0", 7, 7)])
    assert state.dropped == 1 and state.current() == {}


# --- the policy through a real session ----------------------------------------------

def run(readings_by_step: list[list[PieceReading]]) -> Session:
    session = Session(spec_4x3())
    for i, readings in enumerate(readings_by_step):
        place(session, 2.0 * i, readings)
    session.finish(2.0 * len(readings_by_step) + 2)
    return session


def board(spec: JigsawSpec, upto: int) -> list[PieceReading]:
    return [reading(s.part.piece, s.pose.col, s.pose.row, s.pose.rot) for s in spec.steps[:upto]]


def codes(session: Session) -> list[tuple[str, str]]:
    return [(a.code, a.target) for a in session.alerts]


def test_a_correct_build_raises_nothing() -> None:
    spec = spec_4x3()
    session = run([board(spec, k) for k in range(1, 13)])
    assert codes(session) == []
    assert all(status == "done" for _, status, _ in session.checklist())


def test_the_right_piece_in_the_wrong_cell_is_a_wrong_position() -> None:
    spec = spec_4x3()
    full = board(spec, 11)
    stray = reading("r1c2", 1, 1)  # r1c2 belongs at column 2
    session = run([board(spec, k) for k in range(1, 12)] + [[*full, stray]])
    assert ("wrong_position", "s12") in codes(session) or ("wrong_part", "s11") in codes(session)


def test_the_wrong_piece_in_a_wanted_cell_is_a_wrong_part_once() -> None:
    spec = spec_4x3()
    prefix = [board(spec, k) for k in range(1, 4)]
    wrong = [*board(spec, 3), reading("r2c3", 3, 0)]  # a corner piece in another corner's cell
    session = run([*prefix, wrong])
    assert codes(session).count(("wrong_part", "s2")) == 1
    # the dependent steps are not billed for the same mistake
    assert not [c for c in codes(session) if c[0] == "out_of_order"]


def test_a_turned_piece_is_a_wrong_rotation() -> None:
    spec = spec_4x3()
    turned = [*board(spec, 11), reading("r1c2", 2, 1, rot=180)]
    session = run([board(spec, k) for k in range(1, 12)] + [turned])
    assert ("wrong_rotation", "s12") in codes(session)


def test_taking_a_wrong_piece_away_withdraws_the_attempt_quietly() -> None:
    spec = spec_4x3()
    wrong = reading("r2c3", 0, 2)  # the bottom-right corner piece in the bottom-left cell (s3)
    steps = [board(spec, 2), [*board(spec, 2), wrong], board(spec, 2)]
    steps += [board(spec, k) for k in range(3, 13)]
    session = run(steps)
    assert codes(session) == [("wrong_part", "s3")]
    assert all(status == "done" for _, status, _ in session.checklist())


def test_several_unmet_dependencies_make_one_alert_naming_them() -> None:
    spec = spec_4x3()
    session = run([board(spec, 4), [*board(spec, 4), reading("r1c1", 1, 1)]])
    ooo = [a for a in session.alerts if a.code == "out_of_order"]
    assert len(ooo) == 1 and "s5" in ooo[0].text and "s10" in ooo[0].text


def test_a_low_confidence_placement_is_a_warning_not_an_alert() -> None:
    spec = spec_4x3()
    early = [*board(spec, 4), reading("r1c1", 1, 1, conf=0.4)]
    session = run([board(spec, 4), early])
    assert [a.severity for a in session.alerts if a.code == "out_of_order"] == ["warning"]
