"""The generic action vocabulary and post-processing (FR-TMP-2, FR-TMP-3).

The recognizer predicts *verbs*, never step ids: `grasp` is `grasp` whether the hand is
on a Jenga block or a jigsaw piece, so one model serves both procedures without
retraining. Which step it was comes from the world-state tracker, not from here.

Frame-level predictions flicker, so they are smoothed before anything reads them:
`mode_filter` takes the majority label in a sliding window and `merge_short` absorbs
runs shorter than a minimum into their neighbours. Both only ever relabel frames; they
never invent a label that was not predicted nearby.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

#: FR-TMP-2, in the order the model's output channels use.
ACTIONS: tuple[str, ...] = ("search", "grasp", "place", "adjust", "remove", "idle")
INDEX = {a: i for i, a in enumerate(ACTIONS)}


def mode_filter(labels: Sequence[int], window: int = 9) -> list[int]:
    """Majority label in a centred window (ties go to the label nearest the centre)."""
    half = window // 2
    out = []
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): i + half + 1]
        counts = Counter(chunk)
        top = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == top]
        out.append(labels[i] if labels[i] in winners else winners[0])
    return out


def merge_short(labels: Sequence[int], min_len: int = 5) -> list[int]:
    """Absorb runs shorter than `min_len` frames into the longer adjacent run."""
    runs: list[list[int]] = []  # [label, length]
    for lab in labels:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lab, n) in enumerate(runs):
            if n >= min_len:
                continue
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            target = max((r for r in (left, right) if r is not None), key=lambda r: r[1])
            target[1] += n
            del runs[i]
            # Neighbours with the same label now touch; fuse them.
            j = 1
            while j < len(runs):
                if runs[j][0] == runs[j - 1][0]:
                    runs[j - 1][1] += runs[j][1]
                    del runs[j]
                else:
                    j += 1
            changed = True
            break
    return [lab for lab, n in runs for _ in range(n)]
