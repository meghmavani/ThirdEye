"""Jenga end to end: RGB-D frames -> slot reading -> lattice -> checker -> alerts (MG-6, MG-7).

The camera is synthetic (rays cast at a block model); the hand counter is scripted.
Everything between them is the code the live loop runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from eval.synth_tower import corner_camera, full, render, with_block, without

from thirdeye.compiler.schema import ConstraintSpec, load_spec
from thirdeye.perception.depth import RecordedRGBD, save_frame
from thirdeye.perception.jenga_pipeline import JengaPipeline
from thirdeye.perception.tower_view import TowerReader
from thirdeye.session import Session

SPEC = Path(__file__).resolve().parents[1] / "specs" / "jenga.json"
CAMS = [corner_camera(-1, -1), corner_camera(+1, +1)]
DT = 0.1


class Game:
    """Drives a pipeline through scripted physical states, with hands as scripted."""

    def __init__(self, cameras=CAMS, noise: float = 2.0):
        spec = load_spec(SPEC)
        assert isinstance(spec, ConstraintSpec)
        self.session = Session(spec)
        self.hands_now = 0
        self.pipe = JengaPipeline(
            self.session,
            TowerReader(cameras),
            hand_counter=lambda frames: self.hands_now,
        )
        self.cameras, self.noise = cameras, noise
        self.t = 0.0
        self.occ = full()
        self._cache: dict[tuple, list] = {}
        self.alerts = []

    def _frames(self):
        key = tuple(map(tuple, self.occ))
        if key not in self._cache:
            self._cache[key] = [render(c, self.occ, noise_mm=self.noise, seed=1) for c in self.cameras]
        return self._cache[key]

    def run(self, seconds: float, hands: int = 0) -> None:
        self.hands_now = hands
        for _ in range(round(seconds / DT)):
            self.alerts += self.pipe.process(self.t, self._frames())
            self.t += DT

    def move(self, src, dst, hands_on_pull: int = 1) -> None:
        """A hand reaches in, pulls, and goes away; then reaches in again and places."""
        self.run(1.0, hands=hands_on_pull)         # hands on the tower
        self.occ = without(self.occ, src)           # the block is out; the hand has left
        self.run(1.5)
        self.run(0.8, hands=1)                      # carried up and set down
        self.occ = with_block(self.occ, dst)
        self.run(1.5)

    def codes(self) -> list[str]:
        return [a.code for a in self.session.alerts]


def test_legal_moves_raise_nothing_and_the_lattice_follows_the_tower() -> None:
    g = Game()
    g.run(1.0)
    g.move((5, 1), (18, 0))
    g.move((6, 2), (18, 1))
    assert g.codes() == []
    assert g.session.state.current()[5] == (1, 0, 1)
    assert g.session.state.current()[18] == (1, 1, 0)
    assert g.session.summary().startswith("jenga_game: 2 moves judged, 0 alerts")


def test_a_pull_from_the_top_complete_layer_is_an_illegal_source() -> None:
    g = Game()
    g.run(1.0)
    g.move((17, 1), (18, 1))
    assert g.codes() == ["illegal_source_layer"]


def test_two_hands_on_the_pull_is_a_two_handed_move() -> None:
    g = Game()
    g.run(1.0)
    g.move((6, 0), (18, 0), hands_on_pull=2)
    assert g.codes() == ["two_handed_move"]


def test_a_block_put_back_into_the_body_is_a_wrong_replacement() -> None:
    g = Game()
    g.run(1.0)
    for src, dst in (((5, 1), (18, 0)), ((6, 2), (18, 1)), ((7, 0), (18, 2))):
        g.move(src, dst)
    assert g.codes() == []
    g.move((8, 1), (5, 1))
    assert g.codes() == ["wrong_replacement"]


def test_a_missed_hand_detection_does_not_become_a_two_handed_accusation() -> None:
    g = Game()
    g.run(1.0)
    g.move((5, 1), (18, 0), hands_on_pull=0)  # the detector saw nothing during the pull
    assert g.codes() == []


def test_the_tower_falling_ends_the_game_with_one_terminal_alert() -> None:
    g = Game()
    g.run(1.0)
    g.move((5, 1), (18, 0))
    g.occ = [[0, 0, 0]]
    g.run(2.0)
    assert g.session.outcome == "game_over"
    assert g.codes() == ["game_over"]


def test_a_hand_across_the_view_for_a_moment_is_not_a_collapse() -> None:
    g = Game()
    g.run(1.0)
    hand = (-150.0, -150.0, 150.0, -30.0, -30.0, 300.0)
    frames = [render(c, g.occ, hand=hand) for c in CAMS]
    for k in range(2):
        g.alerts += g.pipe.process(g.t, frames)
        g.t += DT
    g.run(1.0)
    assert g.session.outcome is None and g.codes() == []


def test_recorded_rgbd_frames_replay_through_the_same_pipeline(tmp_path: Path) -> None:
    g = Game()
    occ_after = without(full(), (5, 1))
    for i, occ in enumerate([full()] * 3 + [occ_after] * 8):
        for c, cam in enumerate(CAMS):
            save_frame(tmp_path / f"cam{c}" / f"{i:04d}.npz", render(cam, occ, t=i * DT))
    sources = [list(RecordedRGBD(tmp_path / f"cam{c}")) for c in range(2)]
    session = Session(load_spec(SPEC))
    pipe = JengaPipeline(session, TowerReader(CAMS))
    for frames in zip(*sources, strict=True):
        pipe.process(frames[0].t, list(frames))
    assert session.state.current()[5] == (1, 0, 1)
    assert np.isclose(sources[0][0].depth.max(), 1.5)


def test_the_recorded_directory_must_not_be_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RecordedRGBD(tmp_path)


def test_the_command_line_replays_recorded_cameras_with_a_saved_rig(tmp_path: Path, capsys) -> None:
    from thirdeye.live_jenga import main, save_rig

    save_rig(tmp_path / "rig.json", CAMS)
    states = [full()] * 6 + [without(full(), (17, 1))] * 8
    states += [with_block(without(full(), (17, 1)), (18, 1))] * 10
    for c, cam in enumerate(CAMS):
        for i, occ in enumerate(states):
            save_frame(tmp_path / f"cam{c}" / f"{i:04d}.npz", render(cam, occ, t=i * DT))
    log = tmp_path / "game.json"
    code = main(["--spec", str(SPEC), "--camera-dir", str(tmp_path / "cam0"),
                 "--camera-dir", str(tmp_path / "cam1"), "--rig", str(tmp_path / "rig.json"),
                 "--no-hands", "--log", str(log)])
    out = capsys.readouterr().out
    assert code == 0 and "illegal_source_layer" in out and log.exists()


def test_serving_the_ui_alongside_a_recorded_replay_does_not_disturb_it(tmp_path: Path, capsys) -> None:
    from thirdeye.live_jenga import main, save_rig

    save_rig(tmp_path / "rig.json", CAMS)
    states = [full()] * 5 + [without(full(), (17, 1))] * 8
    for c, cam in enumerate(CAMS):
        for i, occ in enumerate(states):
            save_frame(tmp_path / f"cam{c}" / f"{i:04d}.npz", render(cam, occ, t=i * DT))
    code = main(["--spec", str(SPEC), "--camera-dir", str(tmp_path / "cam0"),
                 "--camera-dir", str(tmp_path / "cam1"), "--rig", str(tmp_path / "rig.json"),
                 "--no-hands", "--serve", "18765"])
    assert code == 0 and "UI on http://127.0.0.1:18765/" in capsys.readouterr().out
