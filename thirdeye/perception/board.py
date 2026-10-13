"""The overhead view of a jigsaw board: rectify, find pieces, identify them (FR-JIG-2, FR-JIG-3).

    frame --(homography)--> rectified board --(vs empty board)--> occupied cells
                                                 --(piece_match)--> PieceReading per cell

Calibration (`calibrate.py`, markers around the board, units of cells) gives a
homography from pixels to board-cell coordinates, so the frame can be warped to a
top-down picture in which cell (col, row) is a fixed square, whatever the camera
angle. Segmentation is then a per-cell question -- is something on the empty
board here -- answered by differencing against a picture of the empty board
taken at session start. No learned segmenter, and the plain mat plus even
lighting of HW-6 are what make that enough.

A piece is read at the cell it *sits in*, and identified by what it *looks like*
(`piece_match`), which is how a piece in the wrong cell is found at all: it sits
in cell A and is identified as belonging to cell B.

Pieces off the board plane, stacked or tilted, are out of scope (OOS-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from thirdeye.perception.piece_match import PieceMatcher
from thirdeye.state.jigsaw.piece_grid import PieceReading


def rectify(
    frame: np.ndarray, to_cells: np.ndarray, cols: int, rows: int, cell_px: int
) -> np.ndarray:
    """Warp `frame` to a top-down board image, `cell_px` pixels per cell.

    `to_cells` is the pixel -> cell-coordinate homography from the calibrator.
    """
    scale = np.diag([cell_px, cell_px, 1.0])
    return cv2.warpPerspective(frame, scale @ to_cells, (cols * cell_px, rows * cell_px))


@dataclass
class BoardView:
    """Reads a rectified board into per-cell piece readings."""

    matcher: PieceMatcher
    cell_px: int = 64
    #: Mean absolute difference (0-255) from the empty board above which a cell
    #: holds something. Pieces differ from a plain mat by far more than shadows do.
    occupied_diff: float = 22.0
    #: Fraction of each cell, centred, that is looked at -- the seams and the
    #: neighbouring cell's shadow are ignored.
    inner: float = 0.8

    empty: np.ndarray | None = field(default=None, init=False)

    @property
    def cols(self) -> int:
        return self.matcher.cols

    @property
    def rows(self) -> int:
        return self.matcher.rows

    def learn_empty(self, rectified: np.ndarray) -> None:
        """Remember the empty board. Call once at session start, before any piece."""
        self.empty = rectified.astype(np.float32)

    def _inner(self, img: np.ndarray, col: int, row: int) -> np.ndarray:
        p = self.cell_px
        m = int(p * (1 - self.inner) / 2)
        return img[row * p + m:(row + 1) * p - m, col * p + m:(col + 1) * p - m]

    def occupied(self, rectified: np.ndarray) -> list[tuple[int, int]]:
        """Cells holding something, in reading order."""
        if self.empty is None:
            raise RuntimeError("learn_empty() first: segmentation differences against the empty board")
        out = []
        for row in range(self.rows):
            for col in range(self.cols):
                diff = np.abs(
                    self._inner(rectified.astype(np.float32), col, row)
                    - self._inner(self.empty, col, row)
                ).mean()
                if diff >= self.occupied_diff:
                    out.append((col, row))
        return out

    def read(self, rectified: np.ndarray) -> list[PieceReading]:
        """One reading per occupied cell, identified or not (FR-JIG-4)."""
        cells = self.occupied(rectified)
        p = self.cell_px
        patches = [rectified[r * p:(r + 1) * p, c * p:(c + 1) * p] for c, r in cells]
        return [
            PieceReading(
                col=c,
                row=r,
                piece=m.piece,
                rot=m.rot,
                confidence=m.confidence if m.identified else 1.0,
                margin=m.margin,
            )
            for (c, r), m in zip(cells, self.matcher.identify_many(patches), strict=True)
        ]
