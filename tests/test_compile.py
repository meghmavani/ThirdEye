"""The compile loop and its one repair round (FR-CMP-1, FR-CMP-4, FR-CMP-7, IF-SPEC-5).

No test here spends an API call: the model is a scripted stand-in that returns
canned JSON, which is exactly what the loop's correctness depends on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from thirdeye.compiler.compile import (
    ClaudeLLM,
    CompileError,
    check,
    compile_source,
    main,
    output_schema,
)
from thirdeye.compiler.schema import DagSpec, load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


@dataclass
class Scripted:
    """Returns the given outputs in turn and records every prompt."""

    outputs: list[str]
    model: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str, schema: dict) -> str:
        self.prompts.append(user)
        return self.outputs.pop(0)


def _text(name: str) -> str:
    return (SPECS / name).read_text()


def _floating() -> str:
    data = json.loads(_text("example_dag.json"))
    data["steps"][1]["pose"]["x"] = 9      # s2 now hangs in the air
    return json.dumps(data)


def test_a_valid_first_answer_is_accepted_and_stamped() -> None:
    llm = Scripted([_text("example_dag.json")])
    spec, _warnings = compile_source("1. red brick...", "dag", llm, source_name="m.txt")
    assert isinstance(spec, DagSpec) and len(spec.steps) == 2
    assert spec.provenance is not None
    assert spec.provenance.model == "scripted" and spec.provenance.source == "m.txt"
    assert not spec.provenance.repaired
    assert len(llm.prompts) == 1


def test_a_failing_spec_gets_one_repair_round_with_the_problems_listed() -> None:
    llm = Scripted([_floating(), _text("example_dag.json")])
    spec, _ = compile_source("manual", "dag", llm)
    assert spec.provenance is not None and spec.provenance.repaired
    assert "floating" in llm.prompts[1] and "s2" in llm.prompts[1]
    assert "manual" in llm.prompts[1]


def test_a_second_failure_stops_for_a_human() -> None:
    llm = Scripted([_floating(), _floating(), _text("example_dag.json")])
    with pytest.raises(CompileError) as exc:
        compile_source("manual", "dag", llm)
    assert any("floating" in p for p in exc.value.problems)
    assert exc.value.raw == _floating()
    assert len(llm.prompts) == 2           # never a third call


def test_malformed_output_is_a_problem_not_a_crash() -> None:
    assert check("{not json", "dag").problems[0].startswith("not valid JSON")
    assert check('{"procedure": "x"}', "dag").problems[0].startswith("schema")
    assert "expected a 'dag'" in check(_text("example_constraints.json"), "dag").problems[0]


def test_a_rulebook_compiles_to_the_constraint_form() -> None:
    spec, _ = compile_source("rules", "constraints", Scripted([_text("example_constraints.json")]))
    assert spec.kind == "constraints"


def test_an_illegal_expression_is_caught_by_validation() -> None:
    data = json.loads(_text("example_constraints.json"))
    data["actions"][0]["preconditions"][0]["expr"] = "__import__('os')"
    assert any("bad_expr" in p for p in check(json.dumps(data), "constraints").problems)


def test_the_schema_leaves_provenance_and_prose_out() -> None:
    schema = output_schema("dag")
    assert "provenance" not in schema["properties"]
    assert "Provenance" not in schema.get("$defs", {})
    blob = json.dumps(schema)
    assert "FR-" not in blob and '"title"' not in blob and "prefixItems" not in blob
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_a_compiled_spec_round_trips_through_the_loader(tmp_path: Path) -> None:
    spec, _ = compile_source("m", "dag", Scripted([_text("example_dag.json")]))
    path = tmp_path / "out.json"
    path.write_text(spec.model_dump_json(exclude_none=True))
    assert load_spec(path) == spec


def test_claude_is_asked_for_schema_constrained_json() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="{}")])

    llm = ClaudeLLM(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert llm.complete("sys", "user", {"type": "object"}) == "{}"
    fmt = calls[0]["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": {"type": "object"}}
    assert calls[0]["model"] == "claude-sonnet-5"


def test_a_refusal_is_a_compile_error() -> None:
    def create(**kwargs):
        return SimpleNamespace(stop_reason="refusal", content=[])

    llm = ClaudeLLM(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(CompileError, match="refusal"):
        llm.complete("s", "u", {})


def test_the_command_line_keeps_a_rejected_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "thirdeye.compiler.compile.ClaudeLLM",
        lambda model: Scripted([_floating(), _floating()], model=model),
    )
    src = tmp_path / "manual.txt"
    src.write_text("1. a brick")
    assert main([str(src), "--kind", "dag", "-o", str(tmp_path / "spec.json")]) == 1
    assert (tmp_path / "spec.rejected.json").exists()


# --- the jigsaw kind (FR-CMP-1, FR-CMP-12) ----------------------------------------------

def test_a_jigsaw_sheet_compiles_to_a_piece_grid_spec() -> None:
    from thirdeye.compiler.schema import JigsawSpec

    good = (SPECS / "jigsaw_forest_12.json").read_text()
    llm = Scripted([good])
    spec, _ = compile_source("sheet text", "jigsaw", llm, "manuals/jigsaw_forest_12.txt")
    assert isinstance(spec, JigsawSpec) and len(spec.steps) == 12
    assert spec.provenance is not None and spec.provenance.model == "scripted"
    assert "r<row>c<col>" in llm.prompts[0] or "sheet text" in llm.prompts[0]


def test_the_wrong_form_for_the_requested_kind_is_a_problem() -> None:
    attempt = check((SPECS / "jigsaw_forest_12.json").read_text(), "dag")
    assert attempt.problems and "jigsaw" in attempt.problems[0]


def test_the_jigsaw_output_schema_is_strict_and_provenance_free() -> None:
    schema = output_schema("jigsaw")
    assert "Provenance" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
