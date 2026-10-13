"""RGB-D frames in, Jenga alerts out (plan 4.5; MG-6, MG-7).

    frames (one per camera) -> hands -> settle -> slot reading -> lattice -> checker
                                                \\-> tower height -> collapse detector

The order carries the same discipline as the stud-grid pipeline, adapted to a game
where the hands *are* the action:

*Nothing is read while a hand is on the tower* (FR-PER-6). The pull happens under
the hand; the reading that matters is the settled tower afterwards, taken once the
view has been clear for `settle_s`. The most hands seen on the tower since the last
reading ride in with it as `hands_in_contact` -- the one-hand rule is about the pull
(FR-CHK-10). If the hand detector saw none during a change, the count is passed as
unknown rather than zero: a missed hand must not become a "two-handed" accusation, and
the rule is then simply not in force for that move (see `TowerLattice.observe`).

*A suspected collapse is withheld.* When the measured height drops sharply the
readings are not passed to the lattice until the drop is confirmed or dismissed, so
a tower that has fallen is reported as one terminal event and a glitch that recovers
is reported as nothing (FR-STA-8).

*Unobserved slots keep their belief* (FR-DEP-2), so a slot hidden by the camera
geometry never becomes an invented pull.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from thirdeye.events import Alert
from thirdeye.perception.depth import RGBDFrame
from thirdeye.perception.pipeline import Timings
from thirdeye.perception.tower_view import TowerReader, TowerUpdate
from thirdeye.session import Session
from thirdeye.state.jenga.collapse import CollapseDetector
from thirdeye.state.jenga.lattice import TowerLattice

#: Counts the hands on the tower across the cameras for one tick, or None when no
#: hand detector is available.
HandCounter = Callable[[list[RGBDFrame | None]], int | None]


@dataclass
class JengaPipeline:
    session: Session
    reader: TowerReader
    hand_counter: HandCounter | None = None
    #: How long the tower must be hand-free before a reading is taken (FR-STA-3's
    #: half second, applied to a tower).
    settle_s: float = 0.5
    timings: Timings = field(default_factory=Timings)

    lattice: TowerLattice = field(init=False)
    collapse: CollapseDetector = field(init=False)
    last: TowerUpdate | None = field(default=None, init=False)
    _clear_since: float | None = field(default=None, init=False)
    _max_hands: int = field(default=0, init=False)
    #: Ticks on which a reading was taken, for the report.
    readings: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        state = self.session.state
        if not isinstance(state, TowerLattice):
            raise TypeError("a Jenga pipeline needs a tower-lattice session")
        self.lattice = state
        self.collapse = CollapseDetector(state)

    def process(self, t: float, frames: list[RGBDFrame | None]) -> list[Alert]:
        """One tick. Returns the alerts it produced."""
        clock = time.perf_counter
        start = clock()

        hands = self.hand_counter(frames) if self.hand_counter else None
        if hands:
            self._max_hands = max(self._max_hands, hands)
            self._clear_since = None
            self.timings.add("total", clock() - start)
            return []
        if self._clear_since is None:
            self._clear_since = t
        if t - self._clear_since < self.settle_s and self.readings > 0:
            self.timings.add("total", clock() - start)
            return []

        t0 = clock()
        update = self.reader.update(frames)
        self.timings.add("slots", clock() - t0)
        self.last = update
        self.readings += 1

        confirmed = self.collapse.feed(t, update.height_mm)
        if self.collapse.suspect and not confirmed:
            self.timings.add("total", clock() - start)
            return []

        count = self._max_hands or None if self.hand_counter else None
        self._max_hands = 0
        t0 = clock()
        self.lattice.observe(update.snapshot, t=t, hands_in_contact=count)
        alerts = self.session.pump()
        self.timings.add("state+check", clock() - t0)
        self.timings.add("total", clock() - start)
        return alerts
