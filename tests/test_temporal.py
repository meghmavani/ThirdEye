"""The step-recognizer track: metrics, post-processing, MS-TCN, data and the occlusion hint (S5/S6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from thirdeye.compiler.schema import load_spec
from thirdeye.events import Event
from thirdeye.session import Session
from thirdeye.temporal import metrics as m
from thirdeye.temporal.actions import ACTIONS, INDEX, merge_short, mode_filter
from thirdeye.temporal.data import hand_features, load_split, save_video
from thirdeye.temporal.fusion import Segment, hints_for_occlusion, to_segments
from thirdeye.temporal.mstcn import MSTCN, predict, train

SPECS = Path(__file__).resolve().parents[1] / "specs"


# --- metrics (hand-computed cases) ---------------------------------------------------

def test_segments_run_length_encode_and_drop_background() -> None:
    assert m.segments("aabbbc") == [("a", 0, 2), ("b", 2, 5), ("c", 5, 6)]
    assert m.segments("aabba", background="a") == [("b", 2, 4)]


def test_frame_accuracy_and_its_length_check() -> None:
    assert m.frame_accuracy("abcd", "abcc") == 0.75
    with pytest.raises(ValueError):
        m.frame_accuracy("ab", "abc")


def test_edit_score_punishes_over_segmentation_that_accuracy_forgives() -> None:
    truth = "aaaaaaaabbbbbbbb"
    flicker = "aaaabaaaabbbbbba"  # right most of the time, but cut into pieces
    assert m.frame_accuracy(flicker, truth) > 0.8
    assert m.edit_score(truth, truth) == 100.0
    assert m.edit_score(flicker, truth) < 50.0
    # a, b vs a, b, a, b, a: distance 3, longest 5
    assert m.edit_score("ababa", "ab") == pytest.approx(100 * (1 - 3 / 5))


def test_f1_at_iou_counts_each_true_segment_once() -> None:
    truth = "aaaaabbbbb"
    assert m.f1_at(truth, truth, 0.5) == 100.0
    shifted = "aaabbbbbbb"  # a overlaps 3/5 = 0.6; b overlaps 5/7
    assert m.f1_at(shifted, truth, 0.5) == 100.0
    assert m.f1_at(shifted, truth, 0.75) == 0.0
    twice = "aaaabaabbb"  # two a segments could both claim the one true a
    assert 0 < m.f1_at(twice, truth, 0.25) < 100


def test_report_has_every_ev_seg_1_number() -> None:
    assert set(m.report("aabb", "aabb")) == {"acc", "edit", "f1@10", "f1@25", "f1@50"}


# --- post-processing ------------------------------------------------------------------

def test_mode_filter_removes_single_frame_flicker_but_keeps_real_changes() -> None:
    noisy = [0] * 10 + [1] + [0] * 10 + [2] * 15
    smooth = mode_filter(noisy, window=5)
    assert smooth[10] == 0 and smooth[-1] == 2 and smooth[20] in (0, 2)


def test_merge_short_absorbs_brief_runs_into_the_longer_neighbour() -> None:
    labels = [0] * 10 + [1] * 2 + [0] * 8 + [2] * 10
    assert merge_short(labels, min_len=5) == [0] * 20 + [2] * 10
    assert merge_short([3] * 4, min_len=5) == [3] * 4  # nothing to merge into


# --- the model -------------------------------------------------------------------------

def test_mstcn_learns_a_separable_synthetic_task_and_outputs_one_logit_set_per_stage() -> None:
    rng = np.random.default_rng(0)
    dim = 8
    means = rng.normal(0, 2.0, size=(len(ACTIONS), dim))

    def make():
        labels = []
        for _ in range(6):
            labels += [int(rng.integers(0, len(ACTIONS)))] * int(rng.integers(20, 40))
        feats = np.stack([means[label] + rng.normal(0, 0.5, dim) for label in labels])
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(labels)

    data = [make() for _ in range(4)]
    torch.manual_seed(0)
    model = MSTCN(dim, len(ACTIONS), stages=2, layers=5, channels=16)
    outs = model(data[0][0].t().unsqueeze(0))
    assert len(outs) == 2 and outs[-1].shape == (1, len(ACTIONS), len(data[0][0]))

    history = train(model, data, epochs=45, lr=3e-3)
    assert history[-1] < history[0] * 0.5
    test_feats, test_labels = make()
    pred = predict(model, test_feats)
    assert m.frame_accuracy(pred, test_labels.tolist()) > 0.85


# --- data ---------------------------------------------------------------------------

def test_features_and_labels_round_trip_with_a_verb_mapping(tmp_path: Path) -> None:
    feats = np.random.default_rng(1).normal(size=(5, 4)).astype(np.float32)
    save_video(tmp_path, "v1", feats, ["pick", "pick", "put", "put", "wait"])
    [(name, f, ids)] = load_split(tmp_path, {"pick": "grasp", "put": "place", "wait": "idle"})
    assert name == "v1" and np.allclose(f, feats)
    assert ids == [INDEX["grasp"]] * 2 + [INDEX["place"]] * 2 + [INDEX["idle"]]


def test_an_unmapped_verb_is_an_error_not_idle(tmp_path: Path) -> None:
    save_video(tmp_path, "v", np.zeros((2, 3), np.float32), ["grasp", "wiggle"])
    with pytest.raises(KeyError, match="wiggle"):
        load_split(tmp_path)


def test_a_length_mismatch_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_video(tmp_path, "v", np.zeros((3, 3), np.float32), ["idle"])


def test_hand_features_have_position_presence_size_and_speed() -> None:
    xy = np.array([[100.0, 50.0], [110.0, 50.0], [np.nan, np.nan]])
    f = hand_features(xy, np.array([40.0, 40.0, np.nan]), (200, 100))
    assert f.shape == (3, 5) and f[0, 2] == 1 and f[2, 2] == 0 and f[1, 4] > 0 and f[2, 4] == 0


# --- fusion --------------------------------------------------------------------------

def test_frames_become_timed_segments() -> None:
    segs = to_segments([INDEX["idle"]] * 10 + [INDEX["place"]] * 5, fps=10, t_start=2.0)
    assert segs == [Segment("idle", 2.0, 3.0), Segment("place", 3.0, 3.5)]


def test_a_place_inside_an_occlusion_becomes_a_hint_that_only_activates_the_step() -> None:
    session = Session(load_spec(SPECS / "jigsaw_forest_12.json"))
    segs = [Segment("grasp", 0.0, 1.0), Segment("place", 1.0, 2.0)]
    hints = hints_for_occlusion(session, segs, occluded=[(0.5, 2.5)])
    assert len(hints) == 1 and hints[0].kind == "STEP_START" and hints[0].source == "temporal"
    assert hints[0].part is not None and str(hints[0].part) == "r0c0"

    for h in hints:
        assert session.submit(h) == []  # a hint never produces an alert (FR-CHK-11)
    assert session.checklist()[0][1] == "active"
    assert all(status == "pending" for _, status, _ in session.checklist()[1:])


def test_a_place_outside_any_occlusion_gives_no_hint_and_a_rulebook_gives_none_either() -> None:
    jig = Session(load_spec(SPECS / "jigsaw_forest_12.json"))
    assert hints_for_occlusion(jig, [Segment("place", 5.0, 6.0)], occluded=[(0.0, 1.0)]) == []
    jenga = Session(load_spec(SPECS / "jenga.json"))
    assert hints_for_occlusion(jenga, [Segment("place", 0.5, 1.0)], occluded=[(0.0, 2.0)]) == []
    assert isinstance(Event(t=0, kind="STEP_START"), Event)
