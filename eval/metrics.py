"""The evaluation metrics of plan section 7 / requirements section 9.

**Error detection** (EV-ERR-1, EV-ERR-3, EV-ERR-4, EV-JEN-1). A run is a list of
planted errors (the labels) and the alerts the system raised. An alert is a true
positive if it names a planted error's code and target, and each planted error is
matched at most once; an alert matching nothing is a false positive and a planted
error nothing matched is a miss. Per error code that gives precision and recall;
over fully correct runs it gives false alerts per run, the number a panel member
actually experiences. Delay is alert time minus the time the error happened.

Only red alerts count as detections. A yellow warning is a suspicion (FR-CHK-9),
reported separately so the soft-warning ablation (EV-AB-6) can say what they bought.

**State accuracy** (EV-STA-1, EV-STA-3, EV-STA-4): slot occupancy against a labelled
lattice, jigsaw placement exactness per step, and identification accuracy with the
unidentified rate reported, never folded into accuracy.

The recall and precision targets are *paired* (plan section 7): a system that alerts
constantly scores recall 1.0, so a recall figure never travels without a precision one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import median

from thirdeye.events import Alert

#: Targets of EV-ERR-1 / EV-JEN-1 / EV-ERR-3 / EV-ERR-4.
PRECISION_TARGET = 0.8
RECALL_TARGET = 0.8
FALSE_ALERTS_PER_CORRECT_RUN = 1.0
MEDIAN_DELAY_S = 2.0


@dataclass(frozen=True)
class Planted:
    """A labelled error: what it is, which step/move it concerns, when it happened."""

    code: str
    target: str = ""
    t: float | None = None


@dataclass
class Run:
    name: str
    planted: list[Planted]
    alerts: list[Alert]

    @property
    def correct(self) -> bool:
        """A run with nothing planted: every red alert in it is a false alarm."""
        return not self.planted


@dataclass
class CodeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    def meets_targets(self) -> bool:
        p, r = self.precision, self.recall
        return p is not None and r is not None and p >= PRECISION_TARGET and r >= RECALL_TARGET


@dataclass
class Detection:
    by_code: dict[str, CodeScore] = field(default_factory=lambda: defaultdict(CodeScore))
    delays_s: list[float] = field(default_factory=list)
    false_alerts_in_correct_runs: int = 0
    correct_runs: int = 0
    warnings: int = 0
    #: Alerts raised on a labelled-error run beyond its labels. Counted as false
    #: positives per code above, and totalled here for the headline.
    extra_alerts: int = 0

    @property
    def false_alerts_per_correct_run(self) -> float | None:
        return (self.false_alerts_in_correct_runs / self.correct_runs) if self.correct_runs else None

    @property
    def median_delay_s(self) -> float | None:
        return median(self.delays_s) if self.delays_s else None

    def overall(self) -> CodeScore:
        total = CodeScore()
        for s in self.by_code.values():
            total.tp += s.tp
            total.fp += s.fp
            total.fn += s.fn
        return total


def _matches(planted: Planted, alert: Alert) -> bool:
    return alert.code == planted.code and (not planted.target or alert.target == planted.target)


def score_detection(runs: Iterable[Run]) -> Detection:
    out = Detection()
    for run in runs:
        red = [a for a in run.alerts if a.severity == "alert"]
        out.warnings += len(run.alerts) - len(red)
        if run.correct:
            out.correct_runs += 1
            out.false_alerts_in_correct_runs += len(red)
        unmatched = list(red)
        for p in run.planted:
            hit = next((a for a in unmatched if _matches(p, a)), None)
            if hit is None:
                out.by_code[p.code].fn += 1
                continue
            unmatched.remove(hit)
            out.by_code[p.code].tp += 1
            if p.t is not None:
                out.delays_s.append(hit.t - p.t)
        for a in unmatched:
            out.by_code[a.code].fp += 1
            if not run.correct:
                out.extra_alerts += 1
    return out


def detection_table(d: Detection) -> str:
    """A markdown table, one row per error code, with the headline numbers below."""
    rows = ["| error | TP | FP | FN | precision | recall | meets 0.8/0.8 |", "| --- | --: | --: | --: | --: | --: | :-: |"]
    for code in sorted(d.by_code):
        s = d.by_code[code]
        rows.append(
            f"| {code} | {s.tp} | {s.fp} | {s.fn} | {_f(s.precision)} | {_f(s.recall)} | "
            f"{'yes' if s.meets_targets() else 'no'} |"
        )
    tot = d.overall()
    rows.append(f"| **all** | {tot.tp} | {tot.fp} | {tot.fn} | {_f(tot.precision)} | "
                f"{_f(tot.recall)} | {'yes' if tot.meets_targets() else 'no'} |")
    rows.append("")
    rows.append(f"false alerts per correct run: {_f(d.false_alerts_per_correct_run)} "
                f"over {d.correct_runs} run(s) (target <= {FALSE_ALERTS_PER_CORRECT_RUN})")
    rows.append(f"median delay: {_f(d.median_delay_s)} s over {len(d.delays_s)} labelled time(s) "
                f"(target < {MEDIAN_DELAY_S} s); yellow warnings: {d.warnings}")
    return "\n".join(rows)


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


# --- state accuracy ------------------------------------------------------------------

def slot_accuracy(
    predicted: Sequence[Sequence[int]], truth: Sequence[Sequence[int]]
) -> float:
    """Fraction of slots whose occupancy matches the labelled lattice (EV-STA-3).

    Layers present in only one of the two count as empty in the other, so a tower
    read too short or too tall is penalised for the slots it got wrong.
    """
    height = max(len(predicted), len(truth))
    width = max((len(r) for r in (*predicted, *truth)), default=0)
    if height == 0 or width == 0:
        return 1.0
    right = 0
    for layer in range(height):
        for s in range(width):
            p = predicted[layer][s] if layer < len(predicted) and s < len(predicted[layer]) else 0
            t = truth[layer][s] if layer < len(truth) and s < len(truth[layer]) else 0
            right += int(bool(p) == bool(t))
    return right / (height * width)


def placement_accuracy(
    placed: dict[str, tuple[int, int, int]], truth: dict[str, tuple[int, int, int]]
) -> float:
    """Exact (cell, rotation) per labelled piece (EV-STA-1): `{piece: (col, row, rot)}`."""
    if not truth:
        return 1.0
    return sum(placed.get(piece) == where for piece, where in truth.items()) / len(truth)


@dataclass(frozen=True)
class Identification:
    correct: int
    wrong: int
    unidentified: int

    @property
    def accuracy_among_identified(self) -> float | None:
        n = self.correct + self.wrong
        return self.correct / n if n else None

    @property
    def unidentified_rate(self) -> float:
        total = self.correct + self.wrong + self.unidentified
        return self.unidentified / total if total else 0.0

    def meets_targets(self) -> bool:
        acc = self.accuracy_among_identified
        return acc is not None and acc >= 0.9 and self.unidentified_rate <= 0.1


def score_identification(
    guesses: Iterable[tuple[str | None, int]], truths: Iterable[tuple[str, int]]
) -> Identification:
    """(piece or None, rot) per patch against (piece, rot) ground truth (EV-STA-4)."""
    correct = wrong = unid = 0
    for (piece, rot), (true_piece, true_rot) in zip(guesses, truths, strict=True):
        if piece is None:
            unid += 1
        elif (piece, rot) == (true_piece, true_rot):
            correct += 1
        else:
            wrong += 1
    return Identification(correct, wrong, unid)
