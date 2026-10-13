"""Identify a jigsaw piece by matching it against the box picture (FR-JIG-3..5).

No detector is trained. The box picture is cut into its `cols x rows` cells; a
piece is identified by finding which cell, turned which way, its image most
resembles. Four rotations per cell, so "which piece" and "turned how far" come out
of one search, and a right piece in the wrong cell (a *position* error) is told
apart from the right cell with the wrong turn (a *rotation* error) simply by
comparing the answer with the cell the piece was found in (FR-JIG-5).

**A match is only trusted if it is clearly better than the runner-up** (FR-JIG-4).
`margin` is the gap between the best candidate and the best *other* candidate --
another cell, or the same cell at another turn, so a picture region that looks the
same upside down cannot yield a confident rotation. A piece below `min_margin`, or
whose best score is itself poor, is returned *unidentified* (`piece is None`); the
tracker places it as "something is here" and the checker never alerts on it.

The descriptor is deliberately plain: the cell shrunk to a small colour image,
zero-mean and unit-length, scored by normalised correlation. It is what survives
lighting changes and a little misregistration, and it is cheap -- all `4 x cells`
references are scored against a patch with one matrix product, which is what keeps a
hand-exit match inside NFR-PERF-13 (100 ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from thirdeye.compiler.jigsaw import piece_id

ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class Match:
    """The matcher's verdict on one patch."""

    #: The piece id, or None if unidentified.
    piece: str | None
    #: Degrees clockwise the piece is turned from upright. Meaningful only when `piece` is set.
    rot: int
    #: Normalised correlation with the best candidate, -1..1.
    score: float
    #: Best minus second-best candidate score.
    margin: float
    #: The best candidate even when it was rejected, for the perception report.
    best: str = ""

    @property
    def identified(self) -> bool:
        return self.piece is not None

    @property
    def confidence(self) -> float:
        """0.5..1 for an identified piece, scaled by how clear the win was.

        The floor is the checker's default alert threshold: a match that only
        just cleared `min_margin` can raise a yellow warning at most.
        """
        return 0.0 if self.piece is None else float(np.clip(0.5 + self.margin, 0.5, 1.0))


def _descriptor(img: np.ndarray, size: int) -> np.ndarray:
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=size / 24)
    v = small.reshape(-1)
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


def _turn(img: np.ndarray, degrees: int) -> np.ndarray:
    """`img` turned clockwise by `degrees` (a multiple of 90)."""
    return np.ascontiguousarray(np.rot90(img, k=-(degrees // 90)))


class PieceMatcher:
    """Matches patches against one puzzle's box picture."""

    def __init__(
        self,
        box: np.ndarray,
        cols: int,
        rows: int,
        size: int = 32,
        min_margin: float = 0.08,
        min_score: float = 0.5,
    ):
        if box.ndim == 2:
            box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        self.cols, self.rows, self.size = cols, rows, size
        self.min_margin, self.min_score = min_margin, min_score
        h, w = box.shape[:2]
        self.labels: list[tuple[str, int]] = []
        rows_: list[np.ndarray] = []
        for row in range(rows):
            for col in range(cols):
                cell = box[row * h // rows:(row + 1) * h // rows,
                           col * w // cols:(col + 1) * w // cols]
                for rot in ROTATIONS:
                    rows_.append(_descriptor(_turn(cell, rot), size))
                    self.labels.append((piece_id(col, row), rot))
        self._refs = np.stack(rows_)  # (4 * cells, D)

    def identify(self, patch: np.ndarray) -> Match:
        return self.identify_many([patch])[0]

    def identify_many(self, patches: list[np.ndarray]) -> list[Match]:
        """One matrix product for all patches (NFR-PERF-13)."""
        if not patches:
            return []
        queries = np.stack([_descriptor(p, self.size) for p in patches])  # (n, D)
        scores = queries @ self._refs.T  # (n, 4 * cells)
        out = []
        for row in scores:
            order = np.argsort(row)[::-1]
            best, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            piece, rot = self.labels[best]
            margin = float(row[best] - row[second]) if len(order) > 1 else 1.0
            trusted = margin >= self.min_margin and row[best] >= self.min_score
            out.append(
                Match(
                    piece=piece if trusted else None,
                    rot=rot if trusted else 0,
                    score=float(row[best]),
                    margin=margin,
                    best=piece,
                )
            )
        return out
