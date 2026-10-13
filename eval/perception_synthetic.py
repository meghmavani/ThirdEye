"""Perception ablations on synthetic scenes (EV-AB-9, EV-AB-10; EV-STA-3, EV-STA-4).

    python -m eval.perception_synthetic [--towers 12] [--puzzles 6]

**Read these as a check that the machinery behaves, not as results.** The scenes are
rendered by this repository from the same geometry the reader assumes, so the Jenga
numbers say nothing about a real LiDAR's noise, and the jigsaw numbers say nothing
about real pieces, glare or hands. The real numbers come from labelled recordings
(DR-JEN-3, DR-JIG-4) through the same functions. What the synthetic run *can* show is
the shape of each ablation: what a second camera adds, what the colour cue resolves,
and what dropping rotation search costs.

Jenga (EV-AB-9, EV-STA-3): slot occupancy accuracy among the slots read, and the fraction
left unobserved, for one corner, the other corner, and both merged, under sensor noise
and a hand in front of one face.

Jigsaw (EV-AB-10, EV-STA-4): identification of pieces placed at random turns, with and
without rotation search; accuracy among identified and the unidentified rate, reported
separately as the requirement asks.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from eval.metrics import score_identification
from eval.synth_jigsaw import box_picture
from eval.synth_tower import corner_camera, full, render
from thirdeye.perception.piece_match import ROTATIONS, PieceMatcher, _turn
from thirdeye.perception.tower_view import SlotReader, Thresholds

CAM1, CAM2 = corner_camera(-1, -1), corner_camera(1, 1)


def _random_tower(rng: np.random.Generator) -> list[list[int]]:
    occ = full()
    for _ in range(int(rng.integers(4, 16))):
        occ[int(rng.integers(0, 17))][int(rng.integers(0, 3))] = 0
    occ += [[int(rng.random() < 0.7) for _ in range(3)]] if rng.random() < 0.5 else []
    if occ[-1] == [0, 0, 0]:
        occ.pop()
    return occ


def jenga_ablation(towers: int = 12, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = {
        "corner A only": ([CAM1], True),
        "corner B only": ([CAM2], True),
        "both corners": ([CAM1, CAM2], True),
        "both, depth only": ([CAM1, CAM2], False),
    }
    totals = {name: [0.0, 0.0] for name in configs}  # accuracy among observed, unobserved fraction
    hand = (-120.0, -120.0, 60.0, -40.0, -40.0, 120.0)  # in front of corner A's faces
    for k in range(towers):
        occ = _random_tower(rng)
        want = [[bool(v) for v in row] for row in occ]
        use_hand = k % 2 == 1
        frames = {
            id(c): render(c, occ, noise_mm=3.0, dropout=0.02, hand=hand if use_hand and c is CAM1 else None,
                          seed=k)
            for c in (CAM1, CAM2)
        }
        for name, (cams, colour) in configs.items():
            reader = SlotReader(cams, layers=22, thresholds=Thresholds(), use_colour=colour)
            reading = reader.read([frames[id(c)] for c in cams])
            got, truth = [], []
            for (layer, s), o in reading.obs.items():
                if layer < len(want) + 1 and o.state != "unobserved":
                    got.append(o.state == "occupied")
                    truth.append(want[layer][s] if layer < len(want) else False)
            totals[name][0] += float(np.mean(np.array(got) == np.array(truth))) if got else 0.0
            totals[name][1] += 1 - reading.observed_fraction()
    return [
        {"config": name, "accuracy_among_observed": a / towers, "unobserved": u / towers}
        for name, (a, u) in totals.items()
    ]


def jigsaw_ablation(puzzles: int = 6, cols: int = 4, rows: int = 3, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = {"with rotation search": [], "without rotation search": []}
    truths: list[tuple[str, int]] = []
    guesses: dict[str, list[tuple[str | None, int]]] = {k: [] for k in out}
    for p in range(puzzles):
        box = box_picture(cols, rows, seed=100 + p)
        h, w = box.shape[0] // rows, box.shape[1] // cols
        matchers = {
            "with rotation search": PieceMatcher(box, cols, rows, rotations=ROTATIONS),
            "without rotation search": PieceMatcher(box, cols, rows, rotations=(0,)),
        }
        for r in range(rows):
            for c in range(cols):
                rot = int(rng.choice(ROTATIONS))
                patch = _turn(box[r * h:(r + 1) * h, c * w:(c + 1) * w], rot)
                patch = np.clip(patch.astype(np.float32) * rng.uniform(0.8, 1.1)
                                + rng.normal(0, 6, patch.shape), 0, 255).astype(np.uint8)
                truths.append((f"r{r}c{c}", rot))
                for name, matcher in matchers.items():
                    m = matcher.identify(patch)
                    guesses[name].append((m.piece, m.rot))
    rows_out = []
    for name in out:
        ident = score_identification(guesses[name], truths)
        rows_out.append({
            "config": name,
            "accuracy_among_identified": ident.accuracy_among_identified,
            "unidentified_rate": ident.unidentified_rate,
            "n": len(truths),
        })
    return rows_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--towers", type=int, default=12)
    ap.add_argument("--puzzles", type=int, default=6)
    args = ap.parse_args(argv)

    print("## Jenga slot reading (synthetic; EV-AB-9)\n")
    print("| config | accuracy among observed | unobserved |\n| --- | --: | --: |")
    for r in jenga_ablation(args.towers):
        print(f"| {r['config']} | {r['accuracy_among_observed']:.3f} | {r['unobserved']:.3f} |")
    print("\n## Jigsaw identification at random turns (synthetic; EV-AB-10)\n")
    print("| config | accuracy among identified | unidentified | n |\n| --- | --: | --: | --: |")
    for r in jigsaw_ablation(args.puzzles):
        acc = r["accuracy_among_identified"]
        print(f"| {r['config']} | {'n/a' if acc is None else f'{acc:.3f}'} | "
              f"{r['unidentified_rate']:.3f} | {r['n']} |")
    print("\nSynthetic scenes: a check on the machinery, not a result (see the module docstring).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
