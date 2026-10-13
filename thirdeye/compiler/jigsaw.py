"""Hand-authoring jigsaw specs (FR-CMP-12, DR-CRP-1).

The compiler turns an instruction sheet into a `JigsawSpec`; this builds the
same object directly, for the one ordering the plan's sheets use -- corners,
then the rest of the border, then the interior -- so ground-truth specs for the
eval corpus and the committed demo spec are generated rather than typed. The
order of groups is stated by the sheet, so each group softly depends on the whole
previous group and on nothing physical (hard_depends_on stays empty).
"""

from __future__ import annotations

from thirdeye.compiler.schema import CellPose, Grid, JigsawSpec, JigsawStep, PieceRef


def piece_id(col: int, row: int) -> str:
    """The piece that belongs in cell (col, row) of the finished picture."""
    return f"r{row}c{col}"


def classify(col: int, row: int, cols: int, rows: int) -> str:
    """`corner`, `edge` or `interior` for a cell."""
    on_col = col in (0, cols - 1)
    on_row = row in (0, rows - 1)
    if on_col and on_row:
        return "corner"
    return "edge" if on_col or on_row else "interior"


def _place(col: int, row: int, cols: int, rows: int) -> str:
    side = []
    if row == 0:
        side.append("top")
    elif row == rows - 1:
        side.append("bottom")
    if col == 0:
        side.append("left")
    elif col == cols - 1:
        side.append("right")
    return "-".join(side) if side else f"column {col + 1}, row {row + 1}"


def border_first(
    procedure: str, cols: int, rows: int, rotations: dict[str, int] | None = None
) -> JigsawSpec:
    """Corners, then the other border pieces, then the interior, in reading order.

    `rotations` names pieces the sheet says go in turned, by piece id.
    """
    rotations = rotations or {}
    groups: dict[str, list[tuple[int, int]]] = {"corner": [], "edge": [], "interior": []}
    for row in range(rows):
        for col in range(cols):
            groups[classify(col, row, cols, rows)].append((col, row))

    steps: list[JigsawStep] = []
    previous: list[str] = []
    for kind, cells in groups.items():
        if not cells:
            continue
        ids = []
        for col, row in cells:
            sid = f"s{len(steps) + 1}"
            pid = piece_id(col, row)
            where = _place(col, row, cols, rows)
            what = {"corner": "corner piece", "edge": "edge piece", "interior": "interior piece"}
            steps.append(
                JigsawStep(
                    id=sid,
                    instruction=f"Place {what[kind]} {pid} at {where}",
                    part=PieceRef(piece=pid),
                    pose=CellPose(col=col, row=row, rot=rotations.get(pid, 0)),  # type: ignore[arg-type]
                    soft_depends_on=list(previous),
                )
            )
            ids.append(sid)
        previous = ids
    return JigsawSpec(procedure=procedure, grid=Grid(cols=cols, rows=rows), steps=steps)
