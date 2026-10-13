"""Compiler accuracy over the eval corpus (EV-CMP-1..3, DR-CRP-1..4, EV-AB-4).

    python -m eval.compiler_eval --oracle                 # sanity: ground truth scores 1.0
    python -m eval.compiler_eval --compiled DIR           # score already-compiled specs
    python -m eval.compiler_eval --claude [--model M]     # compile every source, then score

A compiled spec is compared with the hand-authored ground truth, never by text:

*Dependency-form specs* (jigsaw, LEGO): predicted steps are matched to truth steps by
their part and pose, because step ids are an artefact of ordering. Then

    step accuracy        truth steps found with the right part and pose
    dependency accuracy  matched steps whose hard and soft dependency sets are exactly right

and the errors land in the DR-CRP-3 taxonomy:

    wrong_pose           the right part, in a different pose
    missed_dependency    a truth dependency absent from the prediction
    hallucinated_step    a predicted step that is no truth step
    (missing steps are counted too -- a truth step with no part match at all)

*Constraint specs* (Jenga-style rulebooks): two expressions are the same rule if they
agree on a battery of generated towers and moves, not if the strings match --
`src.layer < top_complete_layer` and `src.layer <= top_complete_layer - 1` are one rule.
Truth preconditions with no behavioural twin are `wrong_constraint`; predicted ones with
no twin in the truth are `hallucinated_constraint`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from thirdeye.checker.evaluate import Environment, EvalError, evaluate
from thirdeye.compiler.schema import (
    ConstraintSpec,
    DagSpec,
    JigsawSpec,
    load_spec,
    parse_spec,
)
from thirdeye.events import Slot
from thirdeye.state.jenga.lattice import TowerLattice

CORPUS = Path(__file__).resolve().parents[1] / "manuals" / "eval_corpus"

Spec = DagSpec | JigsawSpec | ConstraintSpec

TARGET_STEPS = 0.9
TARGET_DEPS = 0.8


@dataclass
class SpecScore:
    name: str
    form: str
    steps: float | None = None
    dependencies: float | None = None
    constraints: float | None = None
    errors: dict[str, int] = field(default_factory=dict)
    failed: str = ""

    def bump(self, kind: str, n: int = 1) -> None:
        if n:
            self.errors[kind] = self.errors.get(kind, 0) + n


# --- dependency-form specs ------------------------------------------------------------

def _key(step) -> tuple:
    return (str(step.part), _pose_key(step.pose))


def _pose_key(pose) -> tuple:
    return tuple(sorted(pose.model_dump().items()))


def score_dag(name: str, pred: DagSpec | JigsawSpec, truth: DagSpec | JigsawSpec) -> SpecScore:
    s = SpecScore(name, "jigsaw" if isinstance(truth, JigsawSpec) else "dag")
    by_part = {str(t.part): t for t in truth.steps}
    pred_by_key = {_key(p): p for p in pred.steps}

    found = 0
    mapping: dict[str, str] = {}  # predicted id -> truth id
    for t in truth.steps:
        p = pred_by_key.get(_key(t))
        if p is not None:
            found += 1
            mapping[p.id] = t.id
        elif any(str(q.part) == str(t.part) for q in pred.steps):
            s.bump("wrong_pose")
        else:
            s.bump("missing_step")
    for p in pred.steps:
        if str(p.part) not in by_part:
            s.bump("hallucinated_step")
    s.steps = found / len(truth.steps) if truth.steps else 1.0

    right = checked = 0
    by_id = {t.id: t for t in truth.steps}
    for p in pred.steps:
        tid = mapping.get(p.id)
        if tid is None:
            continue
        checked += 1
        t = by_id[tid]
        got_soft = {mapping.get(d, d) for d in p.soft_depends_on}
        got_hard = {mapping.get(d, d) for d in p.hard_depends_on}
        ok = got_soft == set(t.soft_depends_on) and got_hard == set(t.hard_depends_on)
        right += ok
        if not ok:
            s.bump("missed_dependency", len((set(t.soft_depends_on) - got_soft)
                                            | (set(t.hard_depends_on) - got_hard)))
            s.bump("extra_dependency", len((got_soft - set(t.soft_depends_on))
                                           | (got_hard - set(t.hard_depends_on))))
    # Steps we could not match count against dependency accuracy too: a step that is
    # missing has, in effect, every one of its dependencies wrong.
    s.dependencies = right / len(truth.steps) if truth.steps else 1.0
    return s


# --- constraint specs -----------------------------------------------------------------

def _situations(n: int = 400, seed: int = 11):
    """Random towers with a move and a hand count, for behavioural comparison."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        layers = rng.randint(3, 20)
        rows = [[1, 1, 1] for _ in range(layers)]
        for _ in range(rng.randint(0, layers)):
            rows[rng.randrange(layers)][rng.randrange(3)] = 0
        if rng.random() < 0.5:  # an unfinished top layer
            rows.append([1 if rng.random() < 0.5 else 0 for _ in range(3)])
            if not any(rows[-1]):
                rows[-1][0] = 1
        lattice = TowerLattice(initial=rows)
        height = len(rows)
        src = Slot(rng.randrange(height), rng.randrange(3))
        dst = Slot(rng.randrange(height + 2), rng.randrange(3))
        hands = rng.choice([0, 1, 1, 2])
        collapse = rng.random() < 0.1
        lattice.collapse_detected = collapse
        out.append((lattice, src, dst, hands))
    return out


_SITUATIONS = None


def _outcomes(expr: str) -> tuple | None:
    global _SITUATIONS
    if _SITUATIONS is None:
        _SITUATIONS = _situations()
    results = []
    for lattice, src, dst, hands in _SITUATIONS:
        env = Environment(funcs={"done": lambda _i: False}).merge(*lattice.predicates()).merge(
            {"src": src, "dst": dst, "hands_in_contact": hands}
        )
        try:
            results.append(bool(evaluate(expr, env)))
        except EvalError:
            results.append(None)
    return tuple(results)


def score_constraints(name: str, pred: ConstraintSpec, truth: ConstraintSpec) -> SpecScore:
    s = SpecScore(name, "constraints")
    pred_rules = [(p.violation, _outcomes(p.expr)) for a in pred.actions for p in a.preconditions]
    pred_rules += [(t.outcome, _outcomes(t.expr)) for t in pred.terminal]
    truth_rules = [(p.violation, _outcomes(p.expr)) for a in truth.actions for p in a.preconditions]
    truth_rules += [(t.outcome, _outcomes(t.expr)) for t in truth.terminal]

    used: set[int] = set()
    right = 0
    for _, outcome in truth_rules:
        hit = next((i for i, (_, o) in enumerate(pred_rules) if o == outcome and i not in used), None)
        if hit is None:
            s.bump("wrong_constraint")
        else:
            used.add(hit)
            right += 1
    s.bump("hallucinated_constraint", len(pred_rules) - len(used))
    s.constraints = right / len(truth_rules) if truth_rules else 1.0
    return s


# --- the corpus -----------------------------------------------------------------------

def entries(corpus: Path = CORPUS) -> list[tuple[str, Path, Path]]:
    """(name, source text, ground-truth spec) for every corpus entry."""
    return [(p.stem, p, p.with_suffix(".json")) for p in sorted(corpus.glob("*.txt"))]


def kind_of(truth: Spec) -> str:
    if isinstance(truth, ConstraintSpec):
        return "constraints"
    return "jigsaw" if isinstance(truth, JigsawSpec) else "dag"


def score_pair(name: str, pred: Spec, truth: Spec) -> SpecScore:
    if isinstance(truth, ConstraintSpec) and isinstance(pred, ConstraintSpec):
        return score_constraints(name, pred, truth)
    if isinstance(truth, (DagSpec, JigsawSpec)) and isinstance(pred, type(truth)):
        return score_dag(name, pred, truth)
    return SpecScore(name, kind_of(truth), failed="wrong spec form")


def score_directory(compiled: Path, corpus: Path = CORPUS) -> list[SpecScore]:
    out = []
    for name, _, truth_path in entries(corpus):
        truth = load_spec(truth_path)
        path = compiled / f"{name}.json"
        if not path.exists():
            out.append(SpecScore(name, kind_of(truth), failed="not compiled"))
            continue
        try:
            pred = parse_spec(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # a spec that does not validate is a failed compile
            out.append(SpecScore(name, kind_of(truth), failed=f"invalid: {str(exc)[:80]}"))
            continue
        out.append(score_pair(name, pred, truth))
    return out


def report(scores: list[SpecScore]) -> str:
    rows = ["| source | form | steps | deps | constraints | errors |", "| --- | --- | --: | --: | --: | --- |"]
    for s in scores:
        if s.failed:
            rows.append(f"| {s.name} | {s.form} | - | - | - | FAILED: {s.failed} |")
            continue
        errs = ", ".join(f"{k} x{v}" for k, v in sorted(s.errors.items())) or "-"
        rows.append(f"| {s.name} | {s.form} | {_f(s.steps)} | {_f(s.dependencies)} | "
                    f"{_f(s.constraints)} | {errs} |")
    dag = [s for s in scores if s.steps is not None]
    con = [s for s in scores if s.constraints is not None]
    rows.append("")
    if dag:
        steps = sum(s.steps for s in dag) / len(dag)
        deps = sum(s.dependencies for s in dag) / len(dag)
        rows.append(f"step accuracy {steps:.2f} (target >= {TARGET_STEPS}), "
                    f"dependency accuracy {deps:.2f} (target >= {TARGET_DEPS}) over {len(dag)} sources")
    if con:
        rows.append(f"constraint accuracy {sum(s.constraints for s in con) / len(con):.2f} "
                    f"over {len(con)} sources")
    failed = sum(1 for s in scores if s.failed)
    if failed:
        rows.append(f"{failed} source(s) failed to compile; they count as zero")
    totals: dict[str, int] = {}
    for s in scores:
        for k, v in s.errors.items():
            totals[k] = totals.get(k, 0) + v
    rows.append("error taxonomy: " + (", ".join(f"{k} {v}" for k, v in sorted(totals.items())) or "none"))
    return "\n".join(rows)


def _f(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _oracle(corpus: Path) -> list[SpecScore]:
    return [score_pair(n, load_spec(t), load_spec(t)) for n, _, t in entries(corpus)]


def _compile_all(corpus: Path, model: str, out: Path) -> None:  # pragma: no cover - spends API calls
    from thirdeye.compiler.compile import ClaudeLLM, CompileError, compile_source

    out.mkdir(parents=True, exist_ok=True)
    llm = ClaudeLLM(model=model)
    for name, text_path, truth_path in entries(corpus):
        kind = kind_of(load_spec(truth_path))
        try:
            spec, _ = compile_source(text_path.read_text(encoding="utf-8"), kind, llm, text_path.name)
        except CompileError as exc:
            print(f"{name}: compile failed ({exc})", file=sys.stderr)
            continue
        (out / f"{name}.json").write_text(spec.model_dump_json(indent=2, exclude_none=True) + "\n",
                                          encoding="utf-8")
        print(f"{name}: compiled")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oracle", action="store_true")
    mode.add_argument("--compiled", type=Path)
    mode.add_argument("--claude", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=Path("compiled_corpus"))
    args = ap.parse_args(argv)

    if args.oracle:
        scores = _oracle(args.corpus)
    elif args.claude:
        from thirdeye.compiler.compile import DEFAULT_MODEL

        _compile_all(args.corpus, args.model or DEFAULT_MODEL, args.out)
        scores = score_directory(args.out, args.corpus)
    else:
        scores = score_directory(args.compiled, args.corpus)
    print(report(scores))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
