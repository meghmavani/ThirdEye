"""Write the planted-error scenarios of plan 5.1 for a DAG spec, as replay scripts.

    python scripts/make_scenarios.py specs/model_a.json scripts/scenarios/

One script per planted error -- skip, swap, wrong colour, wrong size, off by a
stud, rotated, extra brick, removed brick -- plus a correct build. Each carries
an `expect:` block, the alerts a correct checker raises and nothing else, so the
same files are a regression suite (`tests/test_scenarios.py`) and a dry run of
the recording protocol before any bricks are on a plate.

Every step is performed the way a person does it: a hand comes over the plate,
the brick appears when the hand leaves. The tracker's stability filter and
occlusion freeze therefore run exactly as they will live.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from thirdeye.compiler.schema import DagSpec, PartRef, Pose, Step, load_spec

#: Seconds a hand is over the plate per step, and the pause after it.
REACH_S = 1.0
PAUSE_S = 1.5


def brick(part: PartRef, pose: Pose) -> str:
    return f"{part} @ {pose.x},{pose.y},{pose.layer},{pose.rot}"


@dataclass
class Build:
    """A timeline of hand-over-plate placements."""

    keyframes: list[dict] = field(default_factory=list)
    t: float = 0.0

    def place(self, text: str) -> None:
        self.keyframes.append({"t": round(self.t, 2), "hands": True})
        self.keyframes.append({"t": round(self.t + REACH_S, 2), "hands": False, "add": [text]})
        self.t += REACH_S + PAUSE_S

    def remove(self, text: str) -> None:
        self.keyframes.append({"t": round(self.t, 2), "hands": True})
        self.keyframes.append(
            {"t": round(self.t + REACH_S, 2), "hands": False, "remove": [text]}
        )
        self.t += REACH_S + PAUSE_S


def _other_color(spec: DagSpec, step: Step) -> PartRef:
    """A part the same size in a colour no step uses at this size."""
    used = {s.part.color for s in spec.steps if s.part.size == step.part.size}
    for color in ("green", "orange", "pink", "grey", "purple"):
        if color not in used:
            return PartRef(color=color, size=step.part.size)
    raise ValueError("no spare colour")  # pragma: no cover


def _other_size(step: Step) -> PartRef:
    for size in ("1x2", "2x2", "1x4", "2x4", "2x3"):
        if size != step.part.size:
            return PartRef(color=step.part.color, size=size)
    raise ValueError("no spare size")  # pragma: no cover


def _fits(spec: DagSpec, part: PartRef, pose: Pose) -> bool:
    w, h = part.dims
    if pose.rot in (90, 270):
        w, h = h, w
    return pose.x + w <= spec.baseplate.studs_x and pose.y + h <= spec.baseplate.studs_y


def scenarios(spec: DagSpec) -> dict[str, tuple[str, list[dict], list[str]]]:
    """name -> (description, keyframes, expected 'code target' alerts)."""
    steps = spec.steps
    n = len(steps)
    mid = steps[n // 2]
    out: dict[str, tuple[str, list[dict], list[str]]] = {}

    def run(order: Sequence[str | Step]) -> list[dict]:
        b = Build()
        for item in order:
            b.place(item if isinstance(item, str) else brick(item.part, item.pose))
        return b.keyframes

    out["correct"] = ("Every step, in order, exactly as the manual says.", run(list(steps)), [])

    # Skip a middle step whose successor depends on it only by order.
    skip = next(
        s for s in steps[1:-1]
        if not any(s.id in t.hard_depends_on for t in steps)
        and any(s.id in t.soft_depends_on for t in steps)
    )
    after = [t.id for t in steps if skip.id in t.soft_depends_on]
    out["skip"] = (
        f"Step {skip.id} is never done.",
        run([s for s in steps if s.id != skip.id]),
        [f"missed_step {skip.id}"] + [f"out_of_order {t}" for t in after],
    )

    # Swap two adjacent flat steps where the second depends on the first by order only.
    i = next(
        k for k in range(n - 1)
        if steps[k].pose.layer == 0 and steps[k + 1].pose.layer == 0
        and steps[k].id in steps[k + 1].soft_depends_on
        and steps[k].id not in steps[k + 1].hard_depends_on
        and k > 0
    )
    order = list(steps)
    order[i], order[i + 1] = order[i + 1], order[i]
    out["swap"] = (
        f"Steps {steps[i].id} and {steps[i + 1].id} are done in the wrong order.",
        run(order),
        [f"out_of_order {steps[i + 1].id}"],
    )

    wrong = _other_color(spec, mid)
    out["wrong_color"] = (
        f"Step {mid.id} uses a {wrong} instead of a {mid.part}.",
        run([brick(wrong, mid.pose) if s is mid else s for s in steps]),
        [f"wrong_brick {mid.id}"],
    )

    sized = next(s for s in steps if _fits(spec, _other_size(s), s.pose))
    other = _other_size(sized)
    out["wrong_size"] = (
        f"Step {sized.id} uses a {other} instead of a {sized.part}.",
        run([brick(other, s.pose) if s is sized else s for s in steps]),
        [f"wrong_brick {sized.id}"],
    )

    # One stud off, onto free cells, on the last step so nothing rests on it.
    last = steps[-1]
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        x, y = last.pose.x + dx, last.pose.y + dy
        if x >= 0 and y >= 0:
            off = Pose(x=x, y=y, layer=last.pose.layer, rot=last.pose.rot)
            if _fits(spec, last.part, off):
                break
    out["offset"] = (
        f"Step {last.id} is placed one stud off.",
        run([brick(last.part, off) if s is last else s for s in steps]),
        [f"wrong_position {last.id}"],
    )

    turned = next(s for s in reversed(steps) if s.part.dims[0] != s.part.dims[1])
    rot = Pose(x=turned.pose.x, y=turned.pose.y, layer=turned.pose.layer,
               rot=(turned.pose.rot + 90) % 360)  # type: ignore[arg-type]
    out["rotated"] = (
        f"Step {turned.id} is turned 90 degrees.",
        run([brick(turned.part, rot) if s is turned else s for s in steps]),
        [f"wrong_rotation {turned.id}"],
    )

    stray = f"purple_1x2 @ {spec.baseplate.studs_x - 2},{spec.baseplate.studs_y - 1},0,0"
    order2: list[str | Step] = list(steps)
    order2.insert(n // 2, stray)
    out["extra"] = ("A brick no step uses is added mid-build.", run(order2), ["extra_part "])

    b = Build()
    for s in steps:
        b.place(brick(s.part, s.pose))
    b.remove(brick(last.part, last.pose))
    out["removed"] = (
        f"The finished model has step {last.id}'s brick taken off again.",
        b.keyframes,
        [f"removed_part {last.id}", f"missed_step {last.id}"],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, DagSpec):
        ap.error("planted-error scenarios are generated for DAG specs only")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.spec.stem
    for name, (desc, keyframes, expect) in scenarios(spec).items():
        path = args.out / f"{stem}_{name}.yaml"
        rel = Path("..", "..", args.spec.as_posix()) if not args.spec.is_absolute() else args.spec
        body = {"spec": rel.as_posix(), "fps": 10, "expect": sorted(expect),
                "keyframes": keyframes}
        header = (
            f"# {desc}\n# Generated by scripts/make_scenarios.py from {args.spec.as_posix()}; "
            "regenerate rather than edit.\n"
        )
        path.write_text(header + yaml.safe_dump(body, sort_keys=False, width=100))
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
