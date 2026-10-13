"""Validator tests (FR-CMP-6, FR-CMP-3)."""

from thirdeye.compiler.validate import errors, is_valid, validate


def codes(spec) -> set[str]:
    return {p.code for p in validate(spec)}


def test_plan_fixtures_are_clean(dag_spec, constraint_spec):
    assert is_valid(validate(dag_spec)), [str(p) for p in validate(dag_spec)]
    assert is_valid(validate(constraint_spec)), [str(p) for p in validate(constraint_spec)]


def test_unknown_dependency(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].soft_depends_on = ["s99"]
    assert "unknown_dependency" in codes(spec)


def test_cycle(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[0].soft_depends_on = ["s2"]
    assert "cycle" in codes(spec)


def test_self_dependency(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[0].soft_depends_on = ["s1"]
    assert "self_dependency" in codes(spec)


def test_collision(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].pose.layer = 0  # now overlaps s1 on the base layer
    assert "collision" in codes(spec)


def test_out_of_bounds(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[0].pose.x = 15  # a 2x4 at x=15 runs off a 16-wide plate
    assert "out_of_bounds" in codes(spec)


def test_floating_part(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].pose.x = 10  # layer 1, nothing beneath
    spec.steps[1].pose.y = 10
    assert "floating" in codes(spec)


def test_missing_hard_dependency_is_a_warning_not_an_error(dag_spec):
    """FR-CMP-3: geometric support implies a hard dep; a gap is the EV-CMP-2 signal."""
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].hard_depends_on = []
    problems = validate(spec)
    assert "missing_hard_dependency" in {p.code for p in problems}
    assert is_valid(problems)  # warning only -- the spec still runs


def test_bad_expr_is_rejected(constraint_spec):
    spec = constraint_spec.model_copy(deep=True)
    spec.actions[0].preconditions[0].expr = "__import__('os').system('rm -rf /')"
    problems = validate(spec)
    assert "bad_expr" in {p.code for p in problems}
    assert not is_valid(problems)


def test_unconstrained_action_warns(constraint_spec):
    spec = constraint_spec.model_copy(deep=True)
    spec.actions[0].preconditions = []
    assert "unconstrained_action" in codes(spec)


def test_problems_are_sorted_errors_first(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].hard_depends_on = []   # warning
    spec.steps[0].pose.x = 15            # error
    severities = [p.severity for p in validate(spec)]
    assert severities == sorted(severities, key=lambda s: 0 if s == "error" else 1)


def test_errors_helper_filters(dag_spec):
    spec = dag_spec.model_copy(deep=True)
    spec.steps[1].hard_depends_on = []
    assert errors(validate(spec)) == []


def test_a_rulebook_with_a_legal_opening_is_satisfiable(constraint_spec) -> None:
    assert not [p for p in validate(constraint_spec) if p.code == "unsatisfiable"]


def test_a_rulebook_that_forbids_every_move_is_refused(constraint_spec) -> None:
    pre = constraint_spec.actions[0].preconditions[0]
    strict = pre.model_copy(update={"expr": "src.layer < 0"})
    action = constraint_spec.actions[0].model_copy(
        update={"preconditions": [strict, *constraint_spec.actions[0].preconditions[1:]]}
    )
    spec = constraint_spec.model_copy(update={"actions": [action]})
    assert [p.code for p in validate(spec) if p.severity == "error"] == ["unsatisfiable"]
