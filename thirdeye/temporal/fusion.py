"""Turning recognized verbs into checker hints during occlusion (FR-TMP-5).

The state tracker is blind while a hand covers the workspace (FR-PER-6) -- exactly when
a Jenga block is touched and released, or a jigsaw piece is picked up and put back. The
temporal model keeps watching. This module is the narrow, deliberately weak bridge:

    a `place` segment that ends inside an occlusion window
        -> a `STEP_START` hint, source "temporal", for the step expected next

A hint can move a step to `active` and no further. Confirmation still has to come from
the state diff once the hand leaves, because an alert resting on a learned model alone is
forbidden (FR-CHK-11) -- see `Checker._on_step_hint`. And the model predicts verbs, not
steps (FR-TMP-3), so the *which step* comes from the checker's own `next_expected()`,
never from the network.

A rulebook gives hints nothing to attach to (a Jenga `move` has no part identity), so for
Jenga this returns no events and the temporal model's only role there is the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

from thirdeye.events import Event
from thirdeye.session import Session
from thirdeye.temporal.actions import ACTIONS


@dataclass(frozen=True)
class Segment:
    label: str
    t0: float
    t1: float


def to_segments(frame_labels: list[int], fps: float, t_start: float = 0.0) -> list[Segment]:
    """Per-frame class ids -> timed segments, merging runs."""
    out: list[Segment] = []
    start = 0
    for i in range(1, len(frame_labels) + 1):
        if i == len(frame_labels) or frame_labels[i] != frame_labels[start]:
            out.append(Segment(ACTIONS[frame_labels[start]],
                               t_start + start / fps, t_start + i / fps))
            start = i
    return out


def hints_for_occlusion(
    session: Session,
    segments: list[Segment],
    occluded: list[tuple[float, float]],
    confidence: float = 0.6,
) -> list[Event]:
    """`STEP_START` hints for `place` segments ending inside an occlusion window."""
    nxt = session.checker.next_expected()
    if nxt is None or nxt.part is None:
        return []
    out = []
    for seg in segments:
        if seg.label != "place":
            continue
        if any(t0 <= seg.t1 <= t1 for t0, t1 in occluded):
            out.append(
                Event(t=seg.t1, kind="STEP_START", source="temporal", part=nxt.part,
                      confidence=confidence)
            )
    return out
