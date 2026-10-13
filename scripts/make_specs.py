"""Regenerate the hand-authored jigsaw specs under specs/.

    python scripts/make_specs.py

`specs/jigsaw_forest_12.json` is what `manuals/jigsaw_forest_12.txt` should
compile to, built directly so the demo does not depend on an API call
(provenance says `hand`). `specs/jenga.json` is the rulebook's constraint spec,
copied from the format example and re-stamped.
"""

from __future__ import annotations

import json
from pathlib import Path

from thirdeye.compiler.jigsaw import border_first
from thirdeye.compiler.schema import ConstraintSpec, Provenance, parse_spec
from thirdeye.compiler.validate import errors, validate

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, spec) -> None:
    bad = errors(validate(spec))
    if bad:
        raise SystemExit(f"{path.name} does not validate: {bad}")
    path.write_text(spec.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    print(path)


def main() -> int:
    hand = lambda src: Provenance(compiler_version="hand", model="hand", source=src)  # noqa: E731
    forest = border_first("jigsaw_forest_12", 4, 3)
    _write(ROOT / "specs" / "jigsaw_forest_12.json",
           forest.model_copy(update={"provenance": hand("manuals/jigsaw_forest_12.txt")}))
    example = border_first("example_jigsaw", 3, 2)
    _write(ROOT / "specs" / "example_jigsaw.json", example)

    data = json.loads((ROOT / "specs" / "example_constraints.json").read_text(encoding="utf-8"))
    data["procedure"] = "jenga_game"
    jenga = parse_spec(data)
    assert isinstance(jenga, ConstraintSpec)
    _write(ROOT / "specs" / "jenga.json",
           jenga.model_copy(update={"provenance": hand("manuals/jenga_rules.txt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
