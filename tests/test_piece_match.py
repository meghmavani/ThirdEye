"""Piece matching and the board view (FR-JIG-2..5, EV-STA-4)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from eval.synth_jigsaw import (
    MARGIN,
    PX,
    box_picture,
    camera_view,
    flat_board,
    layout,
    true_ids,
)
from thirdeye.compiler.jigsaw import piece_id
from thirdeye.perception.board import BoardView, rectify
from thirdeye.perception.calibrate import Calibrator
from thirdeye.perception.piece_match import PieceMatcher, _turn

COLS, ROWS = 4, 3


@pytest.fixture(scope="module")
def box() -> np.ndarray:
    return box_picture(COLS, ROWS)


@pytest.fixture(scope="module")
def matcher(box) -> PieceMatcher:
    return PieceMatcher(box, COLS, ROWS)


def cell_of(box: np.ndarray, col: int, row: int) -> np.ndarray:
    h, w = box.shape[:2]
    return box[row * h // ROWS:(row + 1) * h // ROWS, col * w // COLS:(col + 1) * w // COLS]


def test_every_cell_is_identified_at_every_turn(box, matcher) -> None:
    for col in range(COLS):
        for row in range(ROWS):
            for rot in (0, 90, 180, 270):
                got = matcher.identify(_turn(cell_of(box, col, row), rot))
                assert (got.piece, got.rot) == (piece_id(col, row), rot), (col, row, rot)
                assert got.margin > matcher.min_margin


def test_identification_survives_noise_and_a_lighting_change(box, matcher) -> None:
    rng = np.random.default_rng(3)
    wrong = 0
    for col in range(COLS):
        for row in range(ROWS):
            patch = cell_of(box, col, row).astype(np.float32) * 0.75 + 10
            patch += rng.normal(0, 8, patch.shape)
            got = matcher.identify(np.clip(patch, 0, 255).astype(np.uint8))
            wrong += got.piece != piece_id(col, row)
    assert wrong == 0


def test_a_plain_patch_is_unidentified_not_guessed(matcher) -> None:
    got = matcher.identify(np.full((PX, PX, 3), 120, np.uint8))
    assert got.piece is None and got.confidence == 0.0


def test_two_identical_cells_make_a_piece_ambiguous_and_so_unidentified() -> None:
    picture = box_picture(COLS, ROWS, seed=11)
    h, w = picture.shape[0] // ROWS, picture.shape[1] // COLS
    picture[0:h, w:2 * w] = picture[0:h, 0:w]  # r0c1 is a copy of r0c0
    m = PieceMatcher(picture, COLS, ROWS)
    assert m.identify(picture[0:h, 0:w]).piece is None


def test_a_rotationally_symmetric_cell_has_no_confident_rotation() -> None:
    picture = box_picture(COLS, ROWS, seed=5)
    h, w = picture.shape[0] // ROWS, picture.shape[1] // COLS
    tile = picture[0:h, 0:w]
    sym = np.maximum.reduce([_turn(tile, k) for k in (0, 90, 180, 270)])
    picture[0:h, 0:w] = sym
    m = PieceMatcher(picture, COLS, ROWS)
    assert m.identify(sym).piece is None


def test_batch_matching_agrees_with_single_and_is_fast(box, matcher) -> None:
    import time

    patches = [cell_of(box, c, r) for r in range(ROWS) for c in range(COLS)]
    singles = [matcher.identify(p) for p in patches]
    start = time.perf_counter()
    many = matcher.identify_many(patches)
    elapsed = time.perf_counter() - start
    assert [m.piece for m in many] == [m.piece for m in singles] == true_ids(COLS, ROWS)
    assert elapsed < 0.1  # NFR-PERF-13, with a wide margin on a test machine


# --- the board view, through the whole camera path -----------------------------------

def calibrated_view(box, placed, hidden=frozenset(), **cam):
    lay = layout(COLS, ROWS)
    cal = Calibrator(lay)
    empty_frame = camera_view(flat_board(box, COLS, ROWS))
    c = cal.update(empty_frame)
    assert c.usable
    # markers sit at cell coordinates (-o .. cols+o); homography maps pixels -> cells
    # in the markers' own frame, with the board's top-left at (0, 0).
    view = BoardView(PieceMatcher(box, COLS, ROWS))
    view.learn_empty(rectify(empty_frame, c.homography, COLS, ROWS, view.cell_px))
    frame = camera_view(flat_board(box, COLS, ROWS, placed, hidden), **cam)
    return view, rectify(frame, c.homography, COLS, ROWS, view.cell_px)


def test_the_board_view_finds_and_names_every_piece_through_a_tilted_camera(box) -> None:
    placed = [(piece_id(c, r), c, r, 0) for r in range(ROWS) for c in range(COLS)]
    view, board = calibrated_view(box, placed)
    got = {(r.col, r.row): (r.piece, r.rot) for r in view.read(board)}
    assert got == {(c, r): (piece_id(c, r), 0) for r in range(ROWS) for c in range(COLS)}


def test_an_empty_board_reads_empty(box) -> None:
    view, board = calibrated_view(box, [])
    assert view.read(board) == []


def test_a_piece_in_the_wrong_cell_is_read_where_it_sits_and_named_for_where_it_belongs(box) -> None:
    placed = [("r0c0", 0, 0, 0), ("r2c3", 1, 1, 0)]
    view, board = calibrated_view(box, placed)
    got = {(r.col, r.row): r.piece for r in view.read(board)}
    assert got == {(0, 0): "r0c0", (1, 1): "r2c3"}


def test_a_turned_piece_reports_its_turn(box) -> None:
    view, board = calibrated_view(box, [("r1c2", 2, 1, 270)])
    [r] = view.read(board)
    assert (r.piece, r.rot) == ("r1c2", 270)


def test_the_view_tolerates_sensor_noise_and_a_dimmer_room(box) -> None:
    placed = [(piece_id(c, r), c, r, 90) for r in range(ROWS) for c in range(COLS)]
    view, board = calibrated_view(box, placed, noise=6.0, gain=0.85)
    got = view.read(board)
    assert len(got) == COLS * ROWS
    assert all(r.piece == piece_id(r.col, r.row) and r.rot == 90 for r in got)


def test_reading_without_an_empty_board_is_refused(box) -> None:
    view = BoardView(PieceMatcher(box, COLS, ROWS))
    with pytest.raises(RuntimeError, match="learn_empty"):
        view.read(np.zeros((ROWS * 64, COLS * 64, 3), np.uint8))


def test_rectify_maps_a_known_cell_to_its_square(box) -> None:
    lay = layout(COLS, ROWS)
    cal = Calibrator(lay)
    placed = [("r0c0", 3, 2, 0)]
    frame = camera_view(flat_board(box, COLS, ROWS, placed))
    c = cal.update(frame)
    board = rectify(frame, c.homography, COLS, ROWS, 64)
    square = board[2 * 64:3 * 64, 3 * 64:4 * 64]
    expected = cv2.resize(cell_of(box, 0, 0), (64, 64), interpolation=cv2.INTER_AREA)
    assert np.abs(square.astype(int) - expected.astype(int)).mean() < 25
    assert MARGIN == 2  # the drawing's margin is part of the fixture's contract
