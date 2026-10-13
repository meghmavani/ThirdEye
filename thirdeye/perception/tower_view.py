"""Reading the Jenga lattice from depth in diagonal-corner mode (FR-DEP-1..4, FR-STA-7).

**The geometry.** A tower is 3 blocks across, 75 x 25 x 15 mm blocks, each layer turned
90 degrees from the one below. Even layers run along x, so their three blocks show
their *ends* on the x = 0 and x = 75 faces; odd layers run along y, ends on y = 0 and
y = 75. A camera at a corner sees two adjacent faces -- one x face and one y face --
so it sees the ends of **every** layer, at about 45 degrees, and a second camera at the
opposite corner sees the other two faces. Each of the 54 slots therefore has up to
two end-on windows, a 25 x 15 mm rectangle on a face (plan 2.2, ASM-4).

**The reading.** Every window is a patch of the face plane at a known 3D position, so
the camera model says exactly which depth pixels fall inside it and what depth a flush
block face must have there. The residual `measured - expected` is the whole signal:

    about 0                 a block's end is there             -> occupied
    clearly positive        looking into an empty tunnel        -> empty
    slightly negative       the block is pushed out toward us   -> occupied, pushed
    strongly negative       a hand or another object in front   -> unobserved
    anything in between     colour cue, else                    -> unobserved

One block class and one reading per slot is why a Jenga failure is a reasoning failure
and not a detection one (plan 1.1). **A slot is never guessed** (FR-DEP-2, FR-DEP-3): a
window the camera cannot see, a patch with too little valid depth, a hand in front, and
two cameras that disagree all give `unobserved`, and `TowerReader` holds the previous
belief for that slot so no event is invented. The checker waits; it does not accuse.

The depth-only decision is the primary path. The colour cue (an empty tunnel is shaded)
only resolves residuals too small to call, and fails safe to `unobserved`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from thirdeye.events import Slot
from thirdeye.perception.camera_model import PinholeCamera
from thirdeye.perception.depth import RGBDFrame
from thirdeye.state.jenga.lattice import BLOCK_MM, STANDARD_LAYERS, SLOTS_PER_LAYER

LENGTH, WIDTH, THICK = BLOCK_MM  # 75, 25, 15

Observed = Literal["occupied", "empty", "unobserved"]


@dataclass(frozen=True)
class Window:
    """One end-on slot window on a tower face."""

    layer: int
    slot: int
    centre: np.ndarray
    normal: np.ndarray
    across: np.ndarray  # unit vector along the window's width, in the face plane


def slot_windows(layers: int, slots: int = SLOTS_PER_LAYER) -> list[Window]:
    """Both end faces of every slot. Even layers show ends on the x faces, odd on y."""
    span = slots * WIDTH
    out = []
    for layer in range(layers):
        z = layer * THICK + THICK / 2
        for slot in range(slots):
            along = slot * WIDTH + WIDTH / 2
            if layer % 2 == 0:
                for x, nx in ((0.0, -1.0), (span, 1.0)):
                    out.append(Window(layer, slot, np.array([x, along, z]),
                                      np.array([nx, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])))
            else:
                for y, ny in ((0.0, -1.0), (span, 1.0)):
                    out.append(Window(layer, slot, np.array([along, y, z]),
                                      np.array([0.0, ny, 0.0]), np.array([1.0, 0.0, 0.0])))
    return out


@dataclass(frozen=True)
class Thresholds:
    """Millimetres of residual, tuned on recorded clips (plan 4.6). Defaults are the
    values the synthetic rig is built around; real sensors will want their own."""

    occupied_tol: float = 6.0
    empty: float = 11.0
    push: float = 4.0
    occluder: float = 35.0
    #: Fraction of a window's samples that must hold a valid depth.
    min_valid: float = 0.5
    #: Faces turned further from the camera than this are not read.
    min_cos: float = 0.35
    #: Fraction of the window, centred, that is sampled; the rim blends with neighbours.
    inner: float = 0.6
    #: Colour cue: luminance ratio to a known-occupied window below / above which an
    #: ambiguous window is called empty / occupied.
    dark_ratio: float = 0.65
    light_ratio: float = 0.85


@dataclass(frozen=True)
class SlotObs:
    state: Observed
    pushed_mm: float = 0.0
    #: True when two windows disagreed (reported as unobserved).
    contradictory: bool = False


@dataclass
class _Plan:
    """Everything about one camera that does not change from frame to frame."""

    windows: list[Window]
    rows: np.ndarray      # (n, s) pixel rows, -1 where outside the image
    cols: np.ndarray
    expected: np.ndarray  # (n, s) forward depth, mm
    inside: np.ndarray    # (n, s) bool
    cosine: np.ndarray    # (n,) face-vs-view cosine


def _plan(camera: PinholeCamera, windows: list[Window], th: Thresholds) -> _Plan:
    kept = []
    for w in windows:
        if camera.view_cosine(w.centre, w.normal) >= th.min_cos:
            kept.append(w)
    du = np.array([-1.0, 0.0, 1.0]) * th.inner * WIDTH / 2
    dv = np.array([-1.0, 0.0, 1.0]) * th.inner * THICK / 2
    pts = np.stack([
        w.centre + a * w.across + np.array([0.0, 0.0, b])
        for w in kept for a in du for b in dv
    ]) if kept else np.zeros((0, 3))
    px, z = camera.project(pts) if len(pts) else (np.zeros((0, 2)), np.zeros(0))
    s = len(du) * len(dv)
    cols = np.round(px[:, 0]).astype(int).reshape(-1, s)
    rows = np.round(px[:, 1]).astype(int).reshape(-1, s)
    inside = (cols >= 0) & (cols < camera.width) & (rows >= 0) & (rows < camera.height)
    return _Plan(
        kept,
        np.where(inside, rows, 0),
        np.where(inside, cols, 0),
        z.reshape(-1, s),
        inside & (z.reshape(-1, s) > 0),
        np.array([camera.view_cosine(w.centre, w.normal) for w in kept]),
    )


@dataclass
class TowerReading:
    """The merged reading of one tick: an observation per slot."""

    layers: int
    slots: int
    obs: dict[tuple[int, int], SlotObs] = field(default_factory=dict)

    def at(self, layer: int, slot: int) -> SlotObs:
        return self.obs.get((layer, slot), SlotObs("unobserved"))

    def unobserved(self) -> set[Slot]:
        return {Slot(layer, s) for (layer, s), o in self.obs.items() if o.state == "unobserved"}

    def observed_fraction(self) -> float:
        total = self.layers * self.slots
        return sum(o.state != "unobserved" for o in self.obs.values()) / total if total else 0.0


class SlotReader:
    """Reads window states from one frame per camera and merges them (FR-DEP-1, FR-DEP-2)."""

    def __init__(
        self,
        cameras: list[PinholeCamera],
        layers: int = STANDARD_LAYERS + 4,
        slots: int = SLOTS_PER_LAYER,
        thresholds: Thresholds | None = None,
        use_colour: bool = True,
    ):
        self.layers, self.slots = layers, slots
        self.th = thresholds or Thresholds()
        self.use_colour = use_colour
        windows = slot_windows(layers, slots)
        self.plans = [_plan(cam, windows, self.th) for cam in cameras]

    def read(self, frames: list[RGBDFrame | None]) -> TowerReading:
        """One frame per camera (None for a dropped one) -> merged slot observations."""
        per_slot: dict[tuple[int, int], list[tuple[Observed, float]]] = {}
        for plan, frame in zip(self.plans, frames, strict=True):
            if frame is None:
                continue
            for w, state, push in self._read_one(plan, frame):
                per_slot.setdefault((w.layer, w.slot), []).append((state, push))
        reading = TowerReading(self.layers, self.slots)
        for layer in range(self.layers):
            for slot in range(self.slots):
                reading.obs[(layer, slot)] = _merge(per_slot.get((layer, slot), []))
        return reading

    def _read_one(self, plan: _Plan, frame: RGBDFrame) -> list[tuple[Window, Observed, float]]:
        th = self.th
        if not plan.windows:
            return []
        depth = frame.depth_mm()
        measured = depth[plan.rows, plan.cols]
        measured = np.where(plan.inside, measured, np.nan)
        valid = np.isfinite(measured)
        delta = np.nanmedian(np.where(valid, measured - plan.expected, np.nan), axis=1) \
            if valid.any() else np.full(len(plan.windows), np.nan)
        frac = valid.mean(axis=1)

        lum = None
        if self.use_colour:
            gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lum = np.where(plan.inside, gray[plan.rows, plan.cols], np.nan).mean(axis=1)

        states: list[Observed] = []
        pushes: list[float] = []
        ambiguous: list[int] = []
        for i, d in enumerate(delta):
            push = 0.0
            if frac[i] < th.min_valid or not np.isfinite(d) or d < -th.occluder:
                st: Observed = "unobserved"
            elif d < -th.push:
                st, push = "occupied", float(-d / max(plan.cosine[i], 0.3))
            elif abs(d) <= th.occupied_tol:
                st = "occupied"
            elif d >= th.empty:
                st = "empty"
            else:
                st = "unobserved"
                ambiguous.append(i)
            states.append(st)
            pushes.append(push)

        if ambiguous and lum is not None:
            sure = [lum[i] for i, s in enumerate(states)
                    if s == "occupied" and np.isfinite(lum[i]) and abs(delta[i]) <= th.occupied_tol]
            if len(sure) >= 3:
                ref = float(np.median(sure))
                for i in ambiguous:
                    if not np.isfinite(lum[i]):
                        continue
                    if lum[i] < th.dark_ratio * ref:
                        states[i] = "empty"
                    elif lum[i] > th.light_ratio * ref:
                        states[i] = "occupied"
        return [(w, s, p) for w, s, p in zip(plan.windows, states, pushes, strict=True)]


def _merge(votes: list[tuple[Observed, float]]) -> SlotObs:
    """Combine one slot's windows. Disagreement is unobserved, never a coin toss."""
    occ = [p for s, p in votes if s == "occupied"]
    emp = [s for s, _ in votes if s == "empty"]
    if occ and emp:
        return SlotObs("unobserved", contradictory=True)
    if occ:
        return SlotObs("occupied", pushed_mm=max(occ))
    if emp:
        return SlotObs("empty")
    return SlotObs("unobserved")


@dataclass
class TowerUpdate:
    """What a tick changed in the reader's belief."""

    snapshot: tuple[tuple[int, ...], ...]
    unobserved: set[Slot]
    contradictions: set[Slot]
    pushed: dict[Slot, float]
    #: Tower height by the generous count (see `TowerReader.update`), mm.
    height_mm: float


class TowerReader:
    """A slot reader plus a belief about the whole tower that survives unobserved slots."""

    def __init__(
        self,
        cameras: list[PinholeCamera],
        layers: int = STANDARD_LAYERS,
        slots: int = SLOTS_PER_LAYER,
        extra_layers: int = 4,
        thresholds: Thresholds | None = None,
        use_colour: bool = True,
    ):
        self.layers, self.slots, self.extra = layers, slots, extra_layers
        self.reader = SlotReader(cameras, layers + extra_layers, slots, thresholds, use_colour)
        #: Standard start: the built tower full, nothing above it.
        self.belief = [[1] * slots if layer < layers else [0] * slots
                       for layer in range(layers + extra_layers)]

    def update(self, frames: list[RGBDFrame | None]) -> TowerUpdate:
        """Read, merge into the belief, and report. Unobserved slots keep their belief."""
        reading = self.reader.read(frames)
        unobs, contra, pushed = set(), set(), {}
        for (layer, slot), o in reading.obs.items():
            if o.state == "occupied":
                self.belief[layer][slot] = 1
                if o.pushed_mm > 0:
                    pushed[Slot(layer, slot)] = o.pushed_mm
            elif o.state == "empty":
                self.belief[layer][slot] = 0
            else:
                unobs.add(Slot(layer, slot))
                if o.contradictory:
                    contra.add(Slot(layer, slot))
        # Height counts a layer as standing if any slot in it is occupied *or* cannot
        # be seen: a hand across the upper layers must not look like a collapse.
        top = -1
        for layer in range(len(self.belief) - 1, -1, -1):
            if any(reading.at(layer, s).state == "occupied" or
                   (reading.at(layer, s).state == "unobserved" and self.belief[layer][s])
                   for s in range(self.slots)):
                top = layer
                break
        snap = tuple(tuple(r) for r in _trim(self.belief))
        return TowerUpdate(snap, unobs, contra, pushed, (top + 1) * THICK)


def _trim(rows: list[list[int]]) -> list[list[int]]:
    """Drop empty layers above the highest standing one: the lattice is as tall as the tower."""
    top = max((i for i, r in enumerate(rows) if any(r)), default=-1)
    return rows[: top + 1]


def tower_region_px(camera: PinholeCamera, height_mm: float = STANDARD_LAYERS * THICK + 60,
                    slots: int = SLOTS_PER_LAYER) -> np.ndarray:
    """The tower's outline in one camera's image, for counting hands on it
    (`hands.hands_in_contact`). A hand within a margin of the stack still counts."""
    span = slots * WIDTH
    m = 15.0
    pts = np.array([[x, y, z] for x in (-m, span + m) for y in (-m, span + m)
                    for z in (0.0, height_mm)])
    px, _ = camera.project(pts)
    hull = cv2.convexHull(px.astype(np.float32))
    return hull.reshape(-1, 2)
