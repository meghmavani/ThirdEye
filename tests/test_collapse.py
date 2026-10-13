"""Collapse detection from tower height (FR-STA-8)."""

from __future__ import annotations

from thirdeye.session import Session
from thirdeye.state.jenga.collapse import LAYER_MM, CollapseDetector
from thirdeye.state.jenga.lattice import TowerLattice, full_tower

TALL = 18 * LAYER_MM


def feed(det: CollapseDetector, heights, t0: float = 0.0, dt: float = 0.1) -> list[bool]:
    return [det.feed(t0 + i * dt, h) for i, h in enumerate(heights)]


def test_normal_play_is_not_a_collapse() -> None:
    det = CollapseDetector(TowerLattice())
    # The tower grows a layer, wobbles a few millimetres, never falls.
    heights = [TALL] * 10 + [TALL + LAYER_MM] * 10 + [TALL + LAYER_MM - 3] * 5
    assert not any(feed(det, heights))
    assert not det.collapsed


def test_a_fallen_tower_is_confirmed_and_timestamped_where_it_fell() -> None:
    lattice = TowerLattice()
    det = CollapseDetector(lattice, confirm_frames=3)
    fired = feed(det, [TALL] * 5 + [3 * LAYER_MM] * 5)
    assert fired.index(True) == 7          # confirmed on the third low frame...
    assert det.t_collapse == 0.5           # ...but dated to the first
    assert lattice.collapse_detected


def test_a_one_frame_glitch_recovers_and_is_forgotten() -> None:
    det = CollapseDetector(TowerLattice())
    heights = [TALL] * 5 + [0.0] + [TALL] * 5
    assert not any(feed(det, heights))
    # The glitch never became part of the reference height.
    assert min(det._recent) == TALL


def test_unmeasurable_frames_do_not_break_a_confirmation() -> None:
    det = CollapseDetector(TowerLattice(), confirm_frames=3)
    fired = feed(det, [TALL] * 5 + [LAYER_MM, None, LAYER_MM, None, LAYER_MM])
    assert fired[-1] is True


def test_a_collapse_fires_once() -> None:
    det = CollapseDetector(TowerLattice(), confirm_frames=1)
    assert sum(feed(det, [TALL] * 5 + [0.0] * 5)) == 1


def test_a_detected_collapse_ends_the_session(constraint_spec) -> None:
    session = Session(constraint_spec)
    assert isinstance(session.state, TowerLattice)
    det = CollapseDetector(session.state, confirm_frames=2)
    feed(det, [TALL] * 5 + [0.0] * 2)
    session.state.observe(full_tower(), t=0.6)
    alerts = session.pump()
    assert [a.code for a in alerts] == ["game_over"]
    assert session.outcome == "game_over"
