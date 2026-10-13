"""The LLM procedure compiler (FR-CMP-1, FR-CMP-4, FR-CMP-5, FR-CMP-7, IF-SPEC-5).

    python -m thirdeye.compiler.compile manuals/model_a.txt --kind dag -o specs/model_a.compiled.json
    python -m thirdeye.compiler.compile manuals/jigsaw_forest_12.txt --kind jigsaw -o ...

Text in, validated spec out, once per procedure and never on the live path
(FR-CMP-10). The model sees only text (FR-CMP-11): the manual or rulebook, the
schema, the fixed predicate vocabulary, and one worked example.

The output is constrained twice. The API's structured-output mode holds the
model to the spec's JSON schema, so a missing field or an invented key cannot
happen; then the same `validate()` a hand-written spec goes through checks what
a schema cannot -- cycles, collisions, floating parts, illegal expressions. If
either fails, the model gets **one** repair round with the exact problems
listed (FR-CMP-7), and if that fails too the compiler stops and says why. A
second automatic round would mostly teach us how persuasive the model is.

The LLM is behind a one-method protocol, so the compile loop -- and the repair
round, which is the part worth testing -- runs in the test suite against a
scripted stand-in and never spends a call.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from thirdeye.compiler.expr import vocabulary_prompt
from thirdeye.compiler.schema import (
    ConstraintSpec,
    DagSpec,
    JigsawSpec,
    Provenance,
    parse_spec,
)
from thirdeye.compiler.validate import Problem, errors, validate

COMPILER_VERSION = "0.1.0"
#: Plan section 3. Overridable per call; the choice is recorded in every spec.
DEFAULT_MODEL = "claude-sonnet-5"

Kind = Literal["dag", "jigsaw", "constraints"]

SPECS = Path(__file__).resolve().parents[2] / "specs"
EXAMPLES: dict[str, Path] = {
    "dag": SPECS / "example_dag.json",
    "jigsaw": SPECS / "example_jigsaw.json",
    "constraints": SPECS / "example_constraints.json",
}


class LLM(Protocol):
    """Anything that can turn a prompt into JSON text matching a schema."""

    model: str

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str: ...


class CompileError(Exception):
    """The spec did not validate after the repair round. Needs a human (FR-CMP-9)."""

    def __init__(self, message: str, problems: list[str], raw: str):
        super().__init__(message)
        self.problems = problems
        self.raw = raw


# --- the schema the model is held to -------------------------------------------------

#: JSON-schema keywords that describe rather than constrain, or that
#: structured outputs may not accept. Bounds are dropped here and enforced by
#: Pydantic on the way back in, so nothing is lost.
_DROP = {"title", "description", "default", "minimum", "maximum", "exclusiveMinimum",
         "exclusiveMaximum", "pattern", "minItems", "maxItems"}


def output_schema(kind: Kind) -> dict[str, Any]:
    """The spec's JSON schema, reduced to what the model should fill in.

    Every property is made required, so the model states each default rather
    than leaving us to guess whether it meant it; `provenance` is removed,
    because that is ours to write.
    """
    model = {"dag": DagSpec, "jigsaw": JigsawSpec, "constraints": ConstraintSpec}[kind]
    schema = _strict(model.model_json_schema())
    schema.get("$defs", {}).pop("Provenance", None)
    return schema


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strict(v) for k, v in node.items() if k not in _DROP}
    if "const" in out:
        out["enum"] = [out.pop("const")]
    if "prefixItems" in out:
        # A pair like expected_duration_s: an array of numbers; Pydantic
        # checks the length when the spec is parsed.
        items = out.pop("prefixItems")
        out["items"] = items[0] if items else {}
    if out.get("type") == "object" and "properties" in out:
        out["properties"].pop("provenance", None)
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


# --- prompts --------------------------------------------------------------------------

_DAG_RULES = """\
You compile a LEGO building manual into a dependency-graph procedure spec.

Coordinates are studs on the baseplate. x runs left to right, y runs from the
builder's edge away from them, and (0, 0) is the bottom-left stud. A pose gives
the brick's lowest-x, lowest-y stud. A part of size "WxH" at rot 0 covers W studs
along x and H studs along y; at rot 90 it covers H along x and W along y. Layer 0
sits on the plate; layer n+1 sits on bricks in layer n.

For every step record two kinds of dependency, independently:
- hard_depends_on: the steps whose bricks physically hold this one up -- every
  step on the layer directly below whose studs this brick covers. Nothing else.
- soft_depends_on: the order the manual states. A numbered manual means each
  step softly depends on the one before it. Every hard dependency is also soft.

Use one step per manual instruction, ids s1, s2, ... in manual order, and copy
each instruction verbatim. Colours and sizes are lower-case, sizes like "2x4".
Mark a step optional only if the manual says it is optional. Use
expected_duration_s [3, 20] unless the manual suggests otherwise.
"""

_JIGSAW_RULES = """You compile a jigsaw puzzle instruction sheet into a dependency-graph procedure spec.

The board is a grid of `cols` x `rows` cells. Cell (col, row) has col counting
left to right and row counting top to bottom from (0, 0), the top-left corner.
Pieces are named by where they belong in the finished picture: the piece for
cell (col, row) is "r<row>c<col>", e.g. the top-right corner of a 4-column puzzle
is "r0c3". Every piece goes in its own cell, rotation 0, unless the sheet says
a piece is placed turned (then 90, 180 or 270, clockwise).

Emit one step per piece, expanding any group instruction ("place the four
corners") into one step per piece, ids s1, s2, ... in sheet order. Copy the
instruction text; for an expanded step name the piece.

Dependencies: a loose jigsaw piece is not physically held up by another, so
hard_depends_on is empty. soft_depends_on records the order the sheet states: if
the sheet says all corners before any edge, each edge step softly depends on
every corner step. Do not add dependencies the sheet does not state. Use
expected_duration_s [3, 40] unless the sheet suggests otherwise.
"""

_CONSTRAINT_RULES = """\
You compile a game rulebook into a constraint-form procedure spec: actions with
preconditions, no fixed order. Each rule the player can break becomes one
precondition, with a machine expression, the rule in plain words as `text`,
and a snake_case `violation` code naming the foul. Ending conditions go in
`terminal`.

Expressions may use only this vocabulary:

{vocabulary}
"""


def _system(kind: Kind) -> str:
    rules = {
        "dag": _DAG_RULES,
        "jigsaw": _JIGSAW_RULES,
        "constraints": _CONSTRAINT_RULES.format(vocabulary=vocabulary_prompt()),
    }[kind]
    example = EXAMPLES[kind].read_text()
    return f"{rules}\nA correctly compiled example, for format only:\n\n{example}"


def _repair_prompt(source: str, raw: str, problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"{source}\n\n---\nYour previous spec for this source was:\n\n{raw}\n\n"
        f"It failed validation:\n{listed}\n\n"
        "Return the whole corrected spec. Change only what these problems require."
    )


# --- the compile loop -----------------------------------------------------------------

@dataclass
class Attempt:
    raw: str
    problems: list[str] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    spec: DagSpec | JigsawSpec | ConstraintSpec | None = None


def check(raw: str, kind: Kind) -> Attempt:
    """Parse and validate one model output. Never raises on bad content."""
    attempt = Attempt(raw=raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        attempt.problems = [f"not valid JSON: {exc}"]
        return attempt
    if isinstance(data, dict):
        data.pop("provenance", None)
    try:
        spec = parse_spec(data)
    except ValidationError as exc:
        attempt.problems = [
            f"schema: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return attempt
    form = "jigsaw" if isinstance(spec, JigsawSpec) else spec.kind
    if form != kind:
        attempt.problems = [f"expected a {kind!r} spec, got {form!r}"]
        return attempt
    problems = validate(spec)
    attempt.problems = [str(p) for p in errors(problems)]
    attempt.warnings = [p for p in problems if p.severity == "warning"]
    attempt.spec = spec
    return attempt


def compile_source(
    source: str, kind: Kind, llm: LLM, source_name: str = ""
) -> tuple[DagSpec | JigsawSpec | ConstraintSpec, list[Problem]]:
    """Compile a manual or rulebook. Returns the spec and its validation warnings.

    Raises `CompileError` if the spec still fails after one repair round.
    """
    system = _system(kind)
    schema = output_schema(kind)
    first = check(llm.complete(system, source, schema), kind)
    final, repaired = first, False
    if first.problems:
        final = check(llm.complete(system, _repair_prompt(source, first.raw, first.problems),
                                   schema), kind)
        repaired = True
    if final.problems or final.spec is None:
        raise CompileError(
            f"spec failed validation after one repair round ({len(final.problems)} problem(s))",
            final.problems,
            final.raw,
        )
    spec = final.spec.model_copy(
        update={
            "provenance": Provenance(
                compiler_version=COMPILER_VERSION,
                model=llm.model,
                source=source_name,
                compiled_at=datetime.now(UTC).isoformat(timespec="seconds"),
                repaired=repaired,
            )
        }
    )
    return spec, final.warnings


# --- the real model -------------------------------------------------------------------

@dataclass
class ClaudeLLM:
    """Claude through the Anthropic SDK (the `llm` extra).

    The key comes from the environment or an `ant auth login` profile, never
    from code or config (NFR-PRIV-2).
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    client: Any = None

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        if self.client is None:
            import anthropic  # the `llm` extra; imported here so nothing else needs it

            self.client = anthropic.Anthropic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise CompileError(f"the model stopped early: {response.stop_reason}", [], "")
        return next(b.text for b in response.content if b.type == "text")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="manual or rulebook, plain text")
    ap.add_argument("--kind", choices=["dag", "jigsaw", "constraints"], required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)

    try:
        spec, warnings = compile_source(
            args.source.read_text(), args.kind, ClaudeLLM(model=args.model),
            source_name=args.source.as_posix(),
        )
    except CompileError as exc:
        print(f"compile failed: {exc}", file=sys.stderr)
        for p in exc.problems:
            print(f"  {p}", file=sys.stderr)
        if exc.raw:
            rejected = args.out.with_suffix(".rejected.json")
            rejected.write_text(exc.raw)
            print(f"last attempt kept at {rejected} for hand repair", file=sys.stderr)
        return 1
    args.out.write_text(spec.model_dump_json(indent=2, exclude_none=True) + "\n")
    for w in warnings:
        print(w)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
