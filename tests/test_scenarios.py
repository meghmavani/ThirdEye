"""Every scripted scenario, replayed end to end through tracker, pairer and checker.

Each script under `scripts/scenarios/` names the alerts a correct checker raises
-- the planted error and nothing else. A spare alert here is a false alarm, the
number NFR-REL-2 budgets; a missing one is a planted error we did not catch.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from thirdeye.replay import Replay, ScriptError, brick, main
from thirdeye.sessionlog import SessionRecord

SCENARIOS = sorted((Path(__file__).resolve().parents[1] / "scripts" / "scenarios").glob("*.yaml"))


def _expected(path: Path) -> Counter[str]:
    return Counter(e.rstrip() for e in yaml.safe_load(path.read_text())["expect"])


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_a_scenario_raises_exactly_its_planted_errors(path: Path) -> None:
    record = Replay.load(path).run()
    got = Counter(f"{a.code} {a.target}".rstrip() for a in record.alerts)
    assert got == _expected(path)
    assert record.perception_misses == []


def test_there_are_scenarios_to_run() -> None:
    names = {p.stem for p in SCENARIOS}
    assert {"model_a_correct", "model_a_skip", "jenga_fouls"} <= names


def test_a_session_log_round_trips(tmp_path: Path) -> None:
    record = Replay.load(SCENARIOS[0]).run()
    back = SessionRecord.read(record.write(tmp_path / "log.json"))
    assert back == record
    assert back.timeline() == record.timeline()


def test_a_log_from_another_format_version_is_refused(tmp_path: Path) -> None:
    record = Replay.load(SCENARIOS[0]).run()
    data = record.to_json() | {"version": 999}
    with pytest.raises(ValueError, match="format 999"):
        SessionRecord.from_json(data)


def test_the_command_line_prints_a_summary_and_writes_a_log(tmp_path: Path, capsys) -> None:
    skip = next(p for p in SCENARIOS if p.stem == "model_a_skip")
    assert main([str(skip), "--log", str(tmp_path / "out.json")]) == 0
    out = capsys.readouterr().out
    assert "missed_step" in out and "10/12 steps done" in out
    assert (tmp_path / "out.json").exists()


def test_a_malformed_brick_is_the_script_s_fault() -> None:
    with pytest.raises(ScriptError):
        brick("a red brick somewhere")
    assert brick("blue_2x2 @ 3,4,1,90 conf=0.3").confidence == 0.3
