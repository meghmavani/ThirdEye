"""The evaluation metrics and the compiler scorer (section 7; DR-CRP-3, EV-CMP-1..3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import compiler_eval as ce
from eval import metrics as m
from thirdeye.compiler.schema import CellPose, ConstraintSpec, JigsawSpec, load_spec
from thirdeye.events import Alert

CORPUS = ce.CORPUS


def alert(code: str, target: str = "", t: float = 0.0, severity: str = "alert") -> Alert:
    return Alert(t=t, code=code, rule="r", text="", target=target, severity=severity)  # type: ignore[arg-type]


# --- detection metrics ---------------------------------------------------------------

def test_precision_and_recall_per_code_with_matching_once() -> None:
    runs = [
        m.Run("a", [m.Planted("out_of_order", "s2", 1.0)], [alert("out_of_order", "s2", 2.5)]),
        m.Run("b", [m.Planted("out_of_order", "s3", 1.0)], []),
        m.Run("c", [m.Planted("wrong_part", "s1")], [alert("wrong_part", "s1"), alert("wrong_part", "s1")]),
    ]
    d = m.score_detection(runs)
    oo, wp = d.by_code["out_of_order"], d.by_code["wrong_part"]
    assert (oo.tp, oo.fp, oo.fn) == (1, 0, 1)
    assert (wp.tp, wp.fp, wp.fn) == (1, 1, 0)  # a second identical alert is not a second hit
    assert d.delays_s == [1.5] and d.median_delay_s == 1.5


def test_false_alerts_per_correct_run_counts_only_red_alerts_in_clean_runs() -> None:
    runs = [
        m.Run("clean1", [], [alert("x"), alert("y")]),
        m.Run("clean2", [], [alert("z", severity="warning")]),
        m.Run("dirty", [m.Planted("x")], [alert("x")]),
    ]
    d = m.score_detection(runs)
    assert d.false_alerts_per_correct_run == 1.0 and d.warnings == 1


def test_a_target_wildcard_matches_any_target() -> None:
    d = m.score_detection([m.Run("r", [m.Planted("extra_part")], [alert("extra_part", "")])])
    assert d.by_code["extra_part"].tp == 1


def test_the_paired_targets_are_both_required() -> None:
    always_alert = m.CodeScore(tp=5, fp=45, fn=0)
    assert always_alert.recall == 1.0 and not always_alert.meets_targets()
    assert m.CodeScore(tp=9, fp=1, fn=1).meets_targets()


def test_the_table_has_a_row_per_code_and_the_headlines() -> None:
    d = m.score_detection([m.Run("r", [m.Planted("a", "s1")], [alert("a", "s1")])])
    text = m.detection_table(d)
    assert "| a | 1 | 0 | 0 | 1.00 | 1.00 | yes |" in text and "false alerts per correct run" in text


# --- state metrics ---------------------------------------------------------------------

def test_slot_accuracy_penalises_a_tower_read_too_tall_or_too_short() -> None:
    truth = [[1, 1, 1], [1, 0, 1]]
    assert m.slot_accuracy(truth, truth) == 1.0
    assert m.slot_accuracy([[1, 1, 1], [1, 1, 1]], truth) == pytest.approx(5 / 6)
    assert m.slot_accuracy([[1, 1, 1]], truth) == pytest.approx(4 / 6)


def test_placement_accuracy_is_exact_cell_and_rotation() -> None:
    truth = {"r0c0": (0, 0, 0), "r0c1": (1, 0, 0)}
    assert m.placement_accuracy({"r0c0": (0, 0, 0), "r0c1": (1, 0, 90)}, truth) == 0.5
    assert m.placement_accuracy({}, {}) == 1.0


def test_identification_reports_the_unidentified_rate_instead_of_hiding_it() -> None:
    ident = m.score_identification(
        [("a", 0), ("b", 0), (None, 0), ("c", 90)],
        [("a", 0), ("x", 0), ("c", 0), ("c", 90)],
    )
    assert (ident.correct, ident.wrong, ident.unidentified) == (2, 1, 1)
    assert ident.unidentified_rate == 0.25 and not ident.meets_targets()


# --- the corpus and the compiler scorer -------------------------------------------------

def test_the_corpus_has_sixteen_entries_across_all_three_forms() -> None:
    entries = ce.entries()
    assert len(entries) == 16
    forms = {ce.kind_of(load_spec(t)) for _, _, t in entries}
    assert forms == {"jigsaw", "dag", "constraints"}
    rulebooks = [n for n, _, t in entries if ce.kind_of(load_spec(t)) == "constraints"]
    assert len(rulebooks) >= 3  # DR-CRP-2


def test_every_ground_truth_spec_validates() -> None:
    from thirdeye.compiler.validate import errors, validate

    for name, text, truth in ce.entries():
        assert text.read_text(encoding="utf-8").strip(), name
        assert not errors(validate(load_spec(truth))), name


def test_the_ground_truth_scores_perfectly_against_itself() -> None:
    scores = ce._oracle(CORPUS)
    assert all(s.steps in (None, 1.0) and s.dependencies in (None, 1.0) and s.constraints in (None, 1.0)
               for s in scores)
    assert not any(s.errors for s in scores)


def _truth(name: str):
    return load_spec(CORPUS / f"{name}.json")


def test_a_piece_in_the_wrong_cell_is_a_wrong_pose() -> None:
    truth = _truth("j03")
    assert isinstance(truth, JigsawSpec)
    pred = truth.model_copy(deep=True)
    pred.steps[0].pose = CellPose(col=1, row=1)
    s = ce.score_dag("j03", pred, truth)
    assert s.errors == {"wrong_pose": 1} and s.steps == pytest.approx(8 / 9)


def test_a_dropped_dependency_is_counted_and_costs_dependency_accuracy() -> None:
    truth = _truth("j03")
    assert isinstance(truth, JigsawSpec)
    pred = truth.model_copy(deep=True)
    pred.steps[-1].soft_depends_on = pred.steps[-1].soft_depends_on[:-1]
    s = ce.score_dag("j03", pred, truth)
    assert s.errors.get("missed_dependency") == 1
    assert s.steps == 1.0 and s.dependencies == pytest.approx(8 / 9)


def test_an_invented_step_is_hallucinated_and_a_missing_one_is_missing() -> None:
    truth = _truth("j03")
    assert isinstance(truth, JigsawSpec)
    pred = truth.model_copy(deep=True)
    extra = pred.steps[0].model_copy(deep=True)
    extra.id = "s99"
    extra.part.piece = "r9c9"
    pred.steps = pred.steps[:-1] + [extra]
    s = ce.score_dag("j03", pred, truth)
    assert s.errors == {"missing_step": 1, "hallucinated_step": 1}


def test_step_ids_do_not_matter_only_the_part_and_pose() -> None:
    truth = _truth("j07")
    assert isinstance(truth, JigsawSpec)
    renamed = json.loads(truth.model_dump_json())
    ren = {s["id"]: f"step{i}" for i, s in enumerate(renamed["steps"])}
    for s in renamed["steps"]:
        s["id"] = ren[s["id"]]
        s["soft_depends_on"] = [ren[d] for d in s["soft_depends_on"]]
    pred = JigsawSpec.model_validate(renamed)
    s = ce.score_dag("j07", pred, truth)
    assert s.steps == 1.0 and s.dependencies == 1.0 and not s.errors


def test_an_equivalent_rewrite_of_a_rule_is_the_same_rule() -> None:
    truth = _truth("r01")
    assert isinstance(truth, ConstraintSpec)
    pred = truth.model_copy(deep=True)
    pred.actions[0].preconditions[0].expr = "src.layer < top_complete_layer"
    assert ce.score_constraints("r01", pred, truth).constraints == 1.0


def test_a_wrong_constraint_and_a_hallucinated_one_are_both_counted() -> None:
    truth = _truth("r01")
    assert isinstance(truth, ConstraintSpec)
    pred = truth.model_copy(deep=True)
    pred.actions[0].preconditions[0].expr = "src.layer <= top_complete_layer"  # off by one
    pred.actions[0].preconditions[1].expr = "hands_in_contact == 1 and src.layer >= 2"
    s = ce.score_constraints("r01", pred, truth)
    assert s.errors["wrong_constraint"] == 2 and s.errors["hallucinated_constraint"] == 2
    assert s.constraints == pytest.approx(3 / 5)


def test_the_form_must_match_and_a_missing_compile_counts_as_zero(tmp_path: Path) -> None:
    (tmp_path / "j01.json").write_text(json.dumps(json.loads((CORPUS / "r01.json").read_text())))
    scores = {s.name: s for s in ce.score_directory(tmp_path)}
    assert scores["j01"].failed == "wrong spec form" and scores["j02"].failed == "not compiled"
    assert "FAILED" in ce.report(list(scores.values()))


def test_the_oracle_command_reports_the_targets(capsys) -> None:
    assert ce.main(["--oracle"]) == 0
    out = capsys.readouterr().out
    assert "step accuracy 1.00" in out and "error taxonomy: none" in out
