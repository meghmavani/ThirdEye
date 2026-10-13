"""Collapse detection (FR-STA-8).

A collapse is the one Jenga event the lattice cannot express -- there is no
occupancy grid for a heap -- and the one that ends the session. It is detected
from the tower's measured height, which the side views give directly: normal
play moves the top by at most one layer at a time, while a collapse takes it
down by many layers inside a single frame. The two are separated by an order of
magnitude, which is why the plan calls this trivially separable.

Two guards keep it from firing on a bad frame rather than a fallen tower:

*The reference height is a median* of the last few frames, so a single glitchy
reading cannot become the "before" a later frame is compared with.

*The drop must hold* for `confirm_frames` consecutive frames. A silhouette
briefly cut short -- a hand across the view, a dropped frame -- recovers on the
next one; a fallen tower does not. The collapse is timestamped at the first
low frame, not the confirming one, so the report's reconstruction of the last
moves is anchored to when it actually happened.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from thirdeye.state.jenga.lattice import BLOCK_MM, TowerLattice

#: A layer is one block thick.
LAYER_MM = BLOCK_MM[2]


@dataclass
class CollapseDetector:
    """Watches tower height and marks the lattice when the tower comes down."""

    lattice: TowerLattice
    #: A drop of at least this many layers in one frame is a collapse candidate.
    #: Normal play changes height by at most one.
    min_drop_layers: float = 4.0
    confirm_frames: int = 3
    history: int = 5

    t_collapse: float | None = field(default=None, init=False)
    _recent: deque[float] = field(init=False)
    _pending: tuple[float, float] | None = field(default=None, init=False)
    _low: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.history)

    @property
    def collapsed(self) -> bool:
        return self.t_collapse is not None

    def feed(self, t: float, height_mm: float | None) -> bool:
        """One frame's measured tower height. True on the frame a collapse is confirmed.

        `None` is a frame where the height could not be measured; it neither
        advances nor resets a pending confirmation.
        """
        if self.collapsed or height_mm is None:
            return False

        if self._pending is None and self._recent:
            reference = median(self._recent)
            if self._fell(reference, height_mm):
                self._pending, self._low = (t, reference), 0

        if self._pending is not None:
            t0, reference = self._pending
            if self._fell(reference, height_mm):
                self._low += 1
                if self._low >= self.confirm_frames:
                    self.t_collapse = t0
                    self.lattice.collapse_detected = True
                    return True
                return False
            # It came back: a glitch, not a collapse. The low frames are
            # discarded so they never pollute the reference.
            self._pending, self._low = None, 0

        self._recent.append(height_mm)
        return False

    def _fell(self, reference: float, height_mm: float) -> bool:
        return reference - height_mm >= self.min_drop_layers * LAYER_MM
