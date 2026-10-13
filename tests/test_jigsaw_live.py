"""Jigsaw end to end: camera frames -> calibration -> matching -> tracker -> checker (MG-9).

The only synthetic part is the camera: a tilted view of a drawn board. Everything
after the pixels is the code the live loop runs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from eval.synth_jigsaw import box_picture, camera_view, flat_board

from thirdeye.compiler.schema import JigsawSpec, load_spec
from thirdeye.live_jigsaw import build_pipeline, main
from thirdeye.session import Session

SPEC = Path(__file__).resolve().parents[1] / "specs" / "jigsaw_forest_12.json"
HOLD = 18  # frames each scene is held, comfortably beyond the 0.5 s stability filter at 30 fps


def frames_for(spec: JigsawSpec, box, scenes: list[list[tuple[str, int, int, int]]]):
    yield camera_view(flat_board(box, spec.grid.cols, spec.grid.rows))
    for placed in scenes:
        img = camera_view(flat_board(box, spec.grid.cols, spec.grid.rows, placed))
        for _ in range(HOLD):
            yield img


def steps_of(spec: JigsawSpec):
    return [(s.part.piece, s.pose.col, s.pose.row, s.pose.rot) for s in spec.steps]


def play(spec: JigsawSpec, box, scenes) -> Session:
    session = Session(spec)
    pipe = build_pipeline(spec, box, hands=False)
    t = 0.0
    for frame in frames_for(spec, box, scenes):
        result = pipe.process(frame, t)
        if result.usable:
            session.state.observe(t, result.readings, hands_present=result.hands_present)
            session.pump()
        t += 1 / 30
    session.finish(t)
    return session


@pytest.fixture(scope="module")
def spec() -> JigsawSpec:
    s = load_spec(SPEC)
    assert isinstance(s, JigsawSpec)
    return s


@pytest.fixture(scope="module")
def box(spec):
    return box_picture(spec.grid.cols, spec.grid.rows)


def test_a_correct_run_ticks_every_step_and_raises_nothing(spec, box) -> None:
    steps = steps_of(spec)
    session = play(spec, box, [steps[:k] for k in range(1, 13)])
    assert [a.code for a in session.alerts] == []
    assert all(status == "done" for _, status, _ in session.checklist())


def test_an_interior_piece_before_the_border_is_called_out_of_order(spec, box) -> None:
    steps = steps_of(spec)
    early = steps[:4] + [steps[10]]
    scenes = [steps[:k] for k in range(1, 5)] + [early]
    session = play(spec, box, scenes)
    assert ("out_of_order", "s11") in [(a.code, a.target) for a in session.alerts]


def test_a_piece_turned_a_quarter_is_called_a_wrong_rotation(spec, box) -> None:
    steps = steps_of(spec)
    pid, c, r, _ = steps[0]
    session = play(spec, box, [[(pid, c, r, 90)]])
    assert ("wrong_rotation", "s1") in [(a.code, a.target) for a in session.alerts]


def test_two_pieces_in_each_others_cells_are_two_wrong_parts(spec, box) -> None:
    steps = steps_of(spec)
    (p1, c1, r1, _), (p2, c2, r2, _) = steps[0], steps[1]
    session = play(spec, box, [[(p1, c2, r2, 0), (p2, c1, r1, 0)]])
    wrong = sorted(a.target for a in session.alerts if a.code == "wrong_part")
    assert wrong == ["s1", "s2"]


def test_the_command_line_runs_a_recorded_sequence(spec, box, tmp_path: Path, capsys) -> None:
    steps = steps_of(spec)
    seq = tmp_path / "frames"
    seq.mkdir()
    n = 0
    for frame in frames_for(spec, box, [steps[:k] for k in range(1, 13)]):
        cv2.imwrite(str(seq / f"{n:04d}.png"), frame)
        n += 1
    cv2.imwrite(str(tmp_path / "box.png"), box)
    log = tmp_path / "run.json"
    code = main(["--spec", str(SPEC), "--box", str(tmp_path / "box.png"),
                 "--video", str(seq / "%04d.png"), "--no-hands", "--no-show", "--log", str(log)])
    out = capsys.readouterr().out
    assert code == 0 and "12/12 steps done" in out and "0 alerts" in out
    assert log.exists()
