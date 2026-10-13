"""Write the compiler eval corpus: manuals/eval_corpus/ (DR-CRP-1..4, plan 5.3).

    python scripts/make_corpus.py

Sixteen short sources for procedures that are never built, each with a hand-authored
ground-truth spec beside it (`NAME.txt` and `NAME.json`):

    j01-j09   jigsaw instruction sheets (dependency form, `piece_grid`)
    r01-r04   rulebooks (constraint form, `tower_lattice`)
    l01-l03   LEGO-style numbered manuals (dependency form, `stud_grid`, hard + soft)

The *texts* are written by hand, below, in different voices and layouts on purpose --
numbered lists, prose, group instructions that must be expanded to one step per piece,
positions in words rather than ids -- because a corpus of one template measures nothing.
The *ground truth* is built from compact definitions so it is exact, then validated; a
ground-truth spec that does not pass the validator is a bug here, not a finding.

Pure text, no recording: the one result that survives whether or not the vision stack
works (DR-CRP-4), and released unconditionally (DR-REL-1).
"""

from __future__ import annotations

import json
from pathlib import Path

from thirdeye.compiler.jigsaw import border_first, chain
from thirdeye.compiler.schema import (
    Baseplate,
    DagSpec,
    PartRef,
    Pose,
    Step,
    load_spec,
    parse_spec,
)
from thirdeye.compiler.validate import errors, validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuals" / "eval_corpus"

# --- jigsaw sheets ---------------------------------------------------------------------

LEGEND = (
    "# Pieces are named by where they belong in the box picture: r<row>c<col>, counting\n"
    "# from r0c0 in the top-left. Each piece goes in its own cell of the board.\n\n"
)

JIGSAW: dict[str, tuple[str, object]] = {
    "j01": (
        "# Teddy -- 4 pieces, 2 across and 2 down.\n" + LEGEND +
        "Put each of the four pieces in its place: r0c0 at the top left, r0c1 at the top\n"
        "right, r1c0 at the bottom left and r1c1 at the bottom right. The order does not\n"
        "matter.\n",
        border_first("j01", 2, 2),
    ),
    "j02": (
        "# Sailboat -- 6 pieces, 3 across and 2 down.\n" + LEGEND +
        "1. Lay the four corners: r0c0, r0c2, r1c0 and r1c2.\n"
        "2. Then the two remaining pieces, the ones between the corners: r0c1 on top and\n"
        "   r1c1 on the bottom.\n",
        border_first("j02", 3, 2),
    ),
    "j03": (
        "# Garden -- 9 pieces, 3 x 3.\n" + LEGEND +
        "Border first, then the middle.\n\n"
        "Step 1. Corners: r0c0, r0c2, r2c0, r2c2.\n"
        "Step 2. Once the corners are down, the other four border pieces: r0c1, r1c0, r1c2, r2c1.\n"
        "Step 3. When the border is finished, the centre piece r1c1 goes last.\n",
        border_first("j03", 3, 3),
    ),
    "j04": (
        "# Harbour -- 12 pieces, 4 columns by 3 rows.\n"
        "# Legend: the top row is r0c0 r0c1 r0c2 r0c3, the middle row r1c0 .. r1c3, the\n"
        "# bottom row r2c0 .. r2c3.\n\n"
        "Start with the four corner pieces: top-left, top-right, bottom-left, bottom-right.\n"
        "After that, complete the rest of the frame: the two pieces between the top\n"
        "corners, the two between the bottom corners, and the single piece on each side\n"
        "(middle row, first and last column). Do not start the picture's middle until the\n"
        "frame is complete. The last two pieces are the middle of the middle row, left\n"
        "one first.\n",
        border_first("j04", 4, 3),
    ),
    "j05": (
        "# Lighthouse -- 16 pieces, 4 x 4.\n" + LEGEND +
        "1) Corners: r0c0, r0c3, r3c0, r3c3.\n"
        "2) Frame, after the corners: top r0c1 r0c2; bottom r3c1 r3c2; left r1c0 r2c0;\n"
        "   right r1c3 r2c3.\n"
        "3) Inside, after the whole frame: r1c1, r1c2, r2c1, r2c2.\n",
        border_first("j05", 4, 4),
    ),
    "j06": (
        "# Market -- 20 pieces, 5 across, 4 down.\n" + LEGEND +
        "Work from the outside in. Place the four corners first. Then do the rest of the\n"
        "edge pieces in any order. Only then fill the six interior pieces (rows 2 and 3,\n"
        "columns 2 to 4 -- that is r1c1, r1c2, r1c3, r2c1, r2c2, r2c3).\n",
        border_first("j06", 5, 4),
    ),
    "j07": (
        "# Meadow -- 9 pieces, read like a page.\n" + LEGEND +
        "Place the pieces row by row, left to right, starting at the top: r0c0, r0c1, r0c2,\n"
        "then r1c0, r1c1, r1c2, then r2c0, r2c1, r2c2. Each piece goes down only after the\n"
        "one before it.\n",
        chain("j07", 3, 3, [(c, r) for r in range(3) for c in range(3)]),
    ),
    "j08": (
        "# Kite -- 9 pieces, 3 x 3, one piece goes in turned.\n" + LEGEND +
        "1. The four corners first (r0c0, r0c2, r2c0, r2c2).\n"
        "2. The four edge pieces (r0c1, r1c0, r1c2, r2c1).\n"
        "3. The centre piece r1c1 last. It is the only one that is not upright: turn it a\n"
        "   quarter turn clockwise (90 degrees) before you set it down.\n",
        border_first("j08", 3, 3, rotations={"r1c1": 90}),
    ),
    "j09": (
        "# Spiral -- 9 pieces, 3 x 3, placed clockwise round the picture.\n" + LEGEND +
        "Begin at the top-left piece and go clockwise round the edge, each piece after the\n"
        "previous: r0c0, r0c1, r0c2, r1c2, r2c2, r2c1, r2c0, r1c0. The centre piece r1c1\n"
        "closes the spiral.\n",
        chain("j09", 3, 3, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1)]),
    ),
}

# --- rulebooks ---------------------------------------------------------------------------


def _rules(name: str, instruction: str, pres: list[tuple[str, str, str]],
           terminal: list[tuple[str, str]], effects=None) -> dict:
    return {
        "procedure": name,
        "kind": "constraints",
        "state_model": "tower_lattice",
        "tower": {"layers": 18, "slots_per_layer": 3},
        "actions": [{
            "id": "move",
            "instruction": instruction,
            "preconditions": [
                {"id": f"c{i + 1}", "expr": expr, "text": text, "violation": code}
                for i, (expr, text, code) in enumerate(pres)
            ],
            "effects": effects or ["occupancy[src] = 0", "occupancy[dst] = 1"],
        }],
        "terminal": [
            {"id": f"t{i + 1}", "expr": expr, "outcome": outcome}
            for i, (expr, outcome) in enumerate(terminal)
        ],
    }


RULEBOOKS: dict[str, tuple[str, dict]] = {
    "r01": (
        "# Classic Jenga, as printed in the box.\n\n"
        "Build the tower: 18 layers of 3 blocks, alternate layers crosswise. Take turns.\n"
        "On your turn take one block from any level except the one directly beneath the\n"
        "highest finished level -- wait: not the highest finished level itself, and nothing\n"
        "above it. Use one hand only at a time. Put the block on the very top, in the\n"
        "pattern of the layers: complete a level of three before you begin the next one. A\n"
        "block may not be pushed back into the middle of the tower. The game is over when\n"
        "the tower falls.\n",
        _rules("r01", "Take one block from below the top complete layer and place it on top", [
            ("src.layer <= top_complete_layer - 1", "Do not take from the top complete layer or above",
             "illegal_source_layer"),
            ("hands_in_contact == 1", "One hand at a time", "two_handed_move"),
            ("dst.layer == top_layer or layer_complete(top_layer)",
             "Finish a layer before starting the next", "layer_started_early"),
            ("dst in legal_top_slots()", "Place only in a top-layer slot", "wrong_replacement"),
        ], [("collapse_detected", "game_over")]),
    ),
    "r02": (
        "# Relaxed Jenga -- house rules for beginners.\n\n"
        "Anyone may use both hands, and you may take from any level as long as it is at\n"
        "least two levels below the top of the tower. Whatever you take goes back on top:\n"
        "into a free space on the top level if it is not yet full, otherwise starting a new\n"
        "level. If the tower falls the round ends.\n",
        _rules("r02", "Take a block from at least two layers below the top and place it on top", [
            ("src.layer <= top_layer - 2", "Take only from at least two layers below the top",
             "illegal_source_layer"),
            ("dst in legal_top_slots()", "Place on the top layer, or start a new one when it is full",
             "wrong_replacement"),
        ], [("collapse_detected", "game_over")]),
    ),
    "r03": (
        "# Strict Jenga -- tournament rules.\n\n"
        "1. Never take a block from the highest finished level or any level above it.\n"
        "2. The bottom two levels are the foundation: do not take from them.\n"
        "3. One hand on the tower, always.\n"
        "4. The block goes on top, in a free place on the top level or, once that level is\n"
        "   finished, in a new level.\n"
        "5. A fall ends the game.\n",
        _rules("r03", "Take a block from the middle of the tower and place it on top", [
            ("src.layer <= top_complete_layer - 1", "Not from the highest finished layer or above",
             "illegal_source_layer"),
            ("src.layer >= 2", "The bottom two layers are the foundation", "protected_base"),
            ("hands_in_contact == 1", "One hand only", "two_handed_move"),
            ("dst in legal_top_slots()", "Place on top, in a legal slot", "wrong_replacement"),
        ], [("collapse_detected", "game_over")]),
    ),
    "r04": (
        "# Tower Tag -- a quick variant.\n\n"
        "You may take any block that lies below the highest full level (not from the full\n"
        "level itself). Use a single hand. Finish the level you are building before you\n"
        "start another. Be careful with the foundation: if anybody pulls the middle block\n"
        "of the bottom level the game ends on the spot, and of course it ends if the tower\n"
        "topples.\n",
        _rules("r04", "Take any block below the highest full layer and place it on top", [
            ("src.layer < top_complete_layer", "Take only from below the highest full layer",
             "illegal_source_layer"),
            ("hands_in_contact == 1", "Use a single hand", "two_handed_move"),
            ("dst.layer == top_layer or layer_complete(top_layer)",
             "Finish the level you are building first", "layer_started_early"),
        ], [("collapse_detected", "game_over"), ("src.layer == 0 and src.slot == 1", "foundation_pulled")]),
    ),
}

# --- LEGO-style manuals ------------------------------------------------------------------


def _lego(name: str, parts: list[tuple[str, str, int, int, int, int]]) -> DagSpec:
    """(color, size, x, y, layer, rot) in build order -> a spec with hard (support) and soft
    (numbered order, plus every hard dependency) dependencies, as the compiler prompt defines."""
    steps: list[Step] = []
    for i, (color, size, x, y, layer, rot) in enumerate(parts):
        part, pose = PartRef(color=color, size=size), Pose(x=x, y=y, layer=layer, rot=rot)  # type: ignore[arg-type]
        probe = Step(id=f"s{i + 1}", instruction="", part=part, pose=pose)
        hard = [
            s.id for s in steps
            if s.pose.layer == layer - 1 and s.footprint & probe.footprint
        ]
        soft = sorted(set(hard) | ({steps[-1].id} if steps else set()), key=lambda s: int(s[1:]))
        steps.append(probe.model_copy(update={"hard_depends_on": hard, "soft_depends_on": soft}))
    return DagSpec(procedure=name, baseplate=Baseplate(studs_x=8, studs_y=8), steps=steps)


def _lego_text(title: str, lines: list[str]) -> str:
    head = (f"# {title}\n# 8x8 baseplate. Coordinates: studs from the bottom-left corner; a\n"
            "# brick's position is its lowest-x, lowest-y stud.\n\n")
    return head + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines)) + "\n"


LEGO: dict[str, tuple[str, DagSpec]] = {
    "l01": (
        _lego_text("Three in a row", [
            "Place a red 2x4 brick on the plate in the bottom-left corner, long side pointing away from you.",
            "Place a blue 2x4 brick beside it, two studs to the right, the same way round.",
            "Place a green 2x4 brick beside that, two studs further right.",
        ]),
        _lego("l01", [("red", "2x4", 0, 0, 0, 0), ("blue", "2x4", 2, 0, 0, 0), ("green", "2x4", 4, 0, 0, 0)]),
    ),
    "l02": (
        _lego_text("Little bridge", [
            "Place a red 2x2 brick at the bottom-left corner of the plate.",
            "Place a yellow 2x2 brick two studs to the right of the red one.",
            "Lay a blue 2x4 brick across the top of both, turned so it runs left to right, "
            "its left end over the red brick.",
        ]),
        _lego("l02", [("red", "2x2", 0, 0, 0, 0), ("yellow", "2x2", 2, 0, 0, 0), ("blue", "2x4", 0, 0, 1, 90)]),
    ),
    "l03": (
        _lego_text("Step pyramid", [
            "Place a white 2x4 brick on the plate in the bottom-left corner, long side away from you.",
            "Place a second white 2x4 brick directly to its right.",
            "Bridge them with a black 2x4 turned sideways so it runs left to right across the "
            "front of both, on top.",
            "Put a green 2x2 brick on top of the left white brick, behind the black one.",
        ]),
        _lego("l03", [("white", "2x4", 0, 0, 0, 0), ("white", "2x4", 2, 0, 0, 0),
                      ("black", "2x4", 0, 0, 1, 90), ("green", "2x2", 0, 2, 1, 0)]),
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries: dict[str, tuple[str, object]] = {}
    entries.update(JIGSAW)
    entries.update({k: (text, spec) for k, (text, spec) in LEGO.items()})
    for k, (text, rules) in RULEBOOKS.items():
        entries[k] = (text, parse_spec(rules))

    for name, (text, spec) in entries.items():
        bad = errors(validate(spec))  # type: ignore[arg-type]
        if bad:
            raise SystemExit(f"ground truth {name} does not validate: {[str(b) for b in bad]}")
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        (OUT / f"{name}.json").write_text(
            json.dumps(json.loads(spec.model_dump_json(exclude_none=True)), indent=2) + "\n",  # type: ignore[attr-defined]
            encoding="utf-8",
        )
        load_spec(OUT / f"{name}.json")  # round-trips
    (OUT / "README.md").write_text(
        "# Compiler eval corpus\n\n"
        "Sixteen short sources never physically built, each with a hand-authored ground-truth\n"
        "spec (`NAME.txt` -> `NAME.json`). Regenerate with `python scripts/make_corpus.py`.\n"
        "Score a set of compiled specs with `python -m eval.compiler_eval --compiled DIR`.\n\n"
        "| ids | form | what they exercise |\n| --- | --- | --- |\n"
        "| j01-j09 | jigsaw sheets (`piece_grid`) | group instructions expanded per piece; "
        "positions in words; chains vs border-first; a rotated piece |\n"
        "| r01-r04 | rulebooks (`tower_lattice`) | the same four rules in different words; "
        "rules that differ from classic Jenga; extra terminal conditions |\n"
        "| l01-l03 | LEGO manuals (`stud_grid`) | hard (support) vs soft (order) dependencies |\n\n"
        "Released unconditionally with the code (DR-REL-1).\n",
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} entries to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
