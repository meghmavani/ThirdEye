"""The session report (FR-EXP-1, FR-EXP-2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from thirdeye.explain.report import ClaudeReporter, build_prompt, clock, explain, main, plain_report
from thirdeye.replay import Replay
from thirdeye.sessionlog import SessionRecord

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


def record(name: str) -> SessionRecord:
    return Replay.load(SCENARIOS / f"{name}.yaml").run()


class Stub:
    model = "stub"

    def __init__(self) -> None:
        self.system = self.user = ""

    def complete(self, system: str, user: str) -> str:
        self.system, self.user = system, user
        return "REPORT"


def test_clock_reads_like_a_person_would() -> None:
    assert clock(190.4) == "03:10" and clock(5) == "00:05"


def test_the_plain_report_names_the_time_rule_and_consequence_of_each_alert() -> None:
    text = plain_report(record("jenga_illegal_source"))
    assert "At 00:" in text and "illegal_source_layer" in text and "rule c1" in text
    assert "Do not take from the top complete layer or above" in text
    assert "1 alert(s)" in text


def test_a_clean_session_says_so() -> None:
    assert "No rule was broken." in plain_report(record("jigsaw_forest_12_correct"))


def test_a_collapse_walks_back_through_the_last_moves() -> None:
    text = plain_report(record("jenga_collapse"))
    assert "the tower fell" in text and "The last moves before the end" in text
    assert "MOVE" in text.split("The last moves before the end")[1]


def test_unfinished_steps_are_listed_for_an_ordered_procedure() -> None:
    text = plain_report(record("jigsaw_forest_12_skip"))
    assert "Never completed" in text and "missed_step" in text


def test_our_failures_are_kept_apart_from_the_players() -> None:
    rec = record("jigsaw_forest_12_correct")
    rec.perception_misses = ["s3"]
    rec.unevaluable = ["c2"]
    text = plain_report(rec)
    assert "On our side, not yours" in text and "s3" in text and "c2" in text
    assert "alert(s)" in text and "0 alert(s)" in text  # not counted among the alerts


def test_the_prompt_is_built_from_the_log_alone_and_marks_our_failures() -> None:
    rec = record("jigsaw_forest_12_interior_early")
    rec.perception_misses = ["s7"]
    prompt = build_prompt(rec)
    assert "out_of_order" in prompt and "ON OUR SIDE" in prompt and "s7" in prompt
    assert "frame" not in prompt.lower() and "pixel" not in prompt.lower()


def test_explain_sends_one_call_and_returns_the_models_text() -> None:
    stub = Stub()
    assert explain(record("jenga_two_handed"), stub) == "REPORT"
    assert "two_handed_move" in stub.user and "never attribute" in stub.system.lower() \
        or "not the player" in stub.system
    assert explain(record("jenga_two_handed")).startswith("Session report")


def test_the_claude_reporter_asks_for_plain_text_and_joins_the_blocks() -> None:
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="a"),
                                            SimpleNamespace(type="text", text="b")])

    reporter = ClaudeReporter(model="m", client=SimpleNamespace(messages=Messages()))
    assert reporter.complete("sys", "usr") == "ab"
    assert calls[0]["model"] == "m" and calls[0]["system"] == "sys"


def test_the_command_line_reads_a_log_file(tmp_path: Path, capsys) -> None:
    path = record("jenga_two_handed").write(tmp_path / "g.json")
    assert main([str(path)]) == 0
    assert "two_handed_move" in capsys.readouterr().out
