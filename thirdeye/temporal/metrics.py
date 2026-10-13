"""Temporal action segmentation metrics (EV-SEG-1).

The standard triple from the segmentation literature, over per-frame label sequences:

*Frame accuracy* -- fraction of frames labelled right.

*Edit score* -- segmental edit distance, 100 * (1 - Levenshtein(pred segments, true
segments) / max length). Penalises over-segmentation, which frame accuracy forgives.

*F1@k* -- segments matched by temporal IoU >= k (k = 0.10, 0.25, 0.50), same-label only,
each true segment matched at most once; the F1 of those matches.

Labels are anything comparable (ints or strings). These are written against the
published definitions (MS-TCN, Farha & Gall 2019) so numbers are comparable with a
benchmark's own table; they are checked here against hand-computed cases only, not
against a reference implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

L = TypeVar("L")


def segments(labels: Sequence[L], background: L | None = None) -> list[tuple[L, int, int]]:
    """Run-length encode to (label, start, end_exclusive), dropping `background` runs."""
    out: list[tuple[L, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels and labels[start] != background:
                out.append((labels[start], start, i))
            start = i
    return out


def frame_accuracy(pred: Sequence[L], truth: Sequence[L]) -> float:
    if len(pred) != len(truth):
        raise ValueError(f"length mismatch: {len(pred)} predicted frames, {len(truth)} true")
    if not truth:
        return 1.0
    return float(np.mean([p == t for p, t in zip(pred, truth, strict=True)]))


def _levenshtein(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_score(pred: Sequence[L], truth: Sequence[L], background: L | None = None) -> float:
    p = [s[0] for s in segments(pred, background)]
    t = [s[0] for s in segments(truth, background)]
    if not p and not t:
        return 100.0
    return 100.0 * (1 - _levenshtein(p, t) / max(len(p), len(t)))


def f1_at(pred: Sequence[L], truth: Sequence[L], iou: float, background: L | None = None) -> float:
    ps, ts = segments(pred, background), segments(truth, background)
    if not ps and not ts:
        return 100.0
    used = [False] * len(ts)
    tp = 0
    for label, s, e in ps:
        best, best_j = 0.0, -1
        for j, (tl, ts_, te) in enumerate(ts):
            if tl != label or used[j]:
                continue
            inter = max(0, min(e, te) - max(s, ts_))
            union = max(e, te) - min(s, ts_)
            score = inter / union if union else 0.0
            if score > best:
                best, best_j = score, j
        if best_j >= 0 and best >= iou:
            used[best_j] = True
            tp += 1
    fp, fn = len(ps) - tp, len(ts) - tp
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 100.0 * 2 * precision * recall / (precision + recall)


def report(pred: Sequence[L], truth: Sequence[L], background: L | None = None) -> dict[str, float]:
    """All of EV-SEG-1 for one video (average over videos for a dataset figure)."""
    return {
        "acc": 100.0 * frame_accuracy(pred, truth),
        "edit": edit_score(pred, truth, background),
        "f1@10": f1_at(pred, truth, 0.10, background),
        "f1@25": f1_at(pred, truth, 0.25, background),
        "f1@50": f1_at(pred, truth, 0.50, background),
    }
