"""The Jenga tower lattice (FR-STA-7, FR-STA-8).

An occupancy grid of `layers x slots_per_layer`, plus the alternating layer
orientation, which is structural rather than observed. Two orthogonal side
cameras give full observability -- every layer is end-on to exactly one view, so
its gaps are directly visible (plan 2.2) -- and this class is what they write
into. Deriving the grid from those views is `perception/`'s job; everything here
is pure state, which is why a Jenga failure is attributable to reasoning rather
than to detection (plan 1.1).

**Preconditions are judged against the state before the move.** This is the one
subtle thing in the file. The tracker freezes while hands occlude the workspace
and updates once they leave, so by the time a move reaches the checker the
tower has already changed. Evaluating "finish a layer before starting the next"
against the *new* state would clear every violation of it: placing the first
block of a fresh layer makes that layer the top layer, so `dst.layer ==
top_layer` becomes true precisely because the illegal move happened. So
`predicates()` binds the tower as it stood before the move, and
`collapse_detected` -- which is about now, not before -- is the sole exception.

"Before the move" is not always "the previous reading". A player deliberates
with the block in hand, so the pull and the placement usually land in separate
readings, and the reading between them shows a tower with a gap and nothing on
top. Judging the placement against that would bill one illegal pull as three
fouls. So the baseline only advances from a reading with no block in hand --
one holding at least as many blocks as the baseline did.
"""

from __future__ import annotations

from collections.abc import Sequence

from thirdeye.events import Event, Slot
from thirdeye.state.base import WorldState

#: Immutable rows, so the previous and current snapshots cannot alias. 54 cells
#: does not need NumPy, and comparing tuples is how `diff` stays under the 1 ms
#: of NFR-PERF-7.
Snapshot = tuple[tuple[int, ...], ...]

#: Jenga blocks are 75 x 25 x 15 mm; three across a layer, each layer turned 90
#: degrees from the one below.
BLOCK_MM = (75.0, 25.0, 15.0)
STANDARD_LAYERS = 18
SLOTS_PER_LAYER = 3


def full_tower(layers: int = STANDARD_LAYERS, slots: int = SLOTS_PER_LAYER) -> Snapshot:
    return tuple(tuple(1 for _ in range(slots)) for _ in range(layers))


def empty_tower(layers: int = STANDARD_LAYERS, slots: int = SLOTS_PER_LAYER) -> Snapshot:
    return tuple(tuple(0 for _ in range(slots)) for _ in range(layers))


def orientation(layer: int) -> int:
    """Degrees this layer is turned. Alternating, so it is derived, not observed."""
    return 0 if layer % 2 == 0 else 90


def top_layer(snap: Snapshot) -> int:
    """Highest layer holding any block, or -1 for an empty table."""
    for i in range(len(snap) - 1, -1, -1):
        if any(snap[i]):
            return i
    return -1


def top_complete_layer(snap: Snapshot) -> int:
    """Highest layer with every slot filled, or -1 if none is."""
    for i in range(len(snap) - 1, -1, -1):
        if all(snap[i]):
            return i
    return -1


def layer_complete(snap: Snapshot, layer: int) -> bool:
    if not 0 <= layer < len(snap):
        return False
    return all(snap[layer])


def occupancy_at(snap: Snapshot, layer: int, slot: int) -> int:
    if not (0 <= layer < len(snap) and 0 <= slot < len(snap[layer])):
        return 0
    return snap[layer][slot]


def legal_top_slots(snap: Snapshot) -> set[Slot]:
    """Where a block may legally be put down.

    The gaps in the top layer while it is unfinished; a whole fresh layer once
    it is. This is the rulebook's "finish a layer before starting the next"
    expressed as a set, and the compiled spec reads it as `dst in
    legal_top_slots()`.
    """
    slots = len(snap[0]) if snap else SLOTS_PER_LAYER
    top = top_layer(snap)
    if top < 0:
        return {Slot(0, s) for s in range(slots)}
    if layer_complete(snap, top):
        return {Slot(top + 1, s) for s in range(slots)}
    return {Slot(top, s) for s in range(slots) if snap[top][s] == 0}


class TowerLattice(WorldState):
    """Slot occupancy for a Jenga tower, and the move events between readings."""

    def __init__(self, initial: Snapshot | None = None, slots_per_layer: int = SLOTS_PER_LAYER):
        self.slots_per_layer = slots_per_layer
        snap = full_tower(slots=slots_per_layer) if initial is None else _freeze(initial)
        self._previous: Snapshot = snap
        #: The tower before the move in progress; see the module docstring.
        self._baseline: Snapshot = snap
        self._current: Snapshot = snap
        self._dirty = False
        self._hands: int | None = None
        #: Set by `collapse.py` from a single-frame drop in tower height
        #: (FR-STA-8). Terminal, and never cleared within a session.
        self.collapse_detected = False

    # --- WorldState -----------------------------------------------------------------

    def current(self) -> Snapshot:
        return self._current

    def before(self) -> Snapshot:
        """The tower a move is judged against; see the module docstring."""
        return self._baseline

    def observe(
        self,
        occupancy: Sequence[Sequence[int]],
        t: float = 0.0,
        hands_in_contact: int | None = None,
    ) -> None:
        """Take a new reading from the side views.

        The tower may grow taller than it started, so the layer count is not
        fixed -- only the slots per layer are.

        `hands_in_contact` is the most hands perception saw on the tower since
        the previous reading. The lattice is read after hands leave, so the
        count has to be carried in from the frames that watched the pull; it
        rides on this reading's removals, which is what the one-hand rule is
        about. None means we did not see, and the rule is then not in force for
        this move rather than presumed satisfied.
        """
        snap = _freeze(occupancy)
        if snap and any(len(row) != self.slots_per_layer for row in snap):
            raise ValueError(f"every layer must have {self.slots_per_layer} slots")
        if _blocks(self._current) >= _blocks(self._baseline):
            self._baseline = self._current
        self._previous = self._current
        self._current = snap
        self._dirty = True
        self._t = t
        self._hands = hands_in_contact

    def diff(self) -> list[Event]:
        """Slot changes since the previous reading (FR-STA-8).

        Removals are emitted before arrivals so that `MovePairer` sees the
        halves of a move in the order it expects.
        """
        if not self._dirty:
            return []
        self._dirty = False
        t = getattr(self, "_t", 0.0)
        height = max(len(self._previous), len(self._current))
        removes, adds = [], []
        for layer in range(height):
            for slot in range(self.slots_per_layer):
                was = occupancy_at(self._previous, layer, slot)
                now = occupancy_at(self._current, layer, slot)
                if was == now:
                    continue
                if was and not now:
                    removes.append(
                        Event(
                            t=t,
                            kind="REMOVE",
                            src=Slot(layer, slot),
                            hands_in_contact=self._hands,
                        )
                    )
                else:
                    adds.append(Event(t=t, kind="ADD", dst=Slot(layer, slot)))
        events = removes + adds
        if self.collapse_detected:
            events.append(Event(t=t, kind="COLLAPSE"))
        return events

    def predicates(self):
        """Bind the rulebook's vocabulary to the tower before the move."""
        snap = self._baseline
        names = {
            "top_layer": top_layer(snap),
            "top_complete_layer": top_complete_layer(snap),
            "collapse_detected": self.collapse_detected,
        }
        funcs = {
            "layer_complete": lambda layer: layer_complete(snap, layer),
            "legal_top_slots": lambda: legal_top_slots(snap),
            "occupancy": lambda layer, slot: occupancy_at(snap, layer, slot),
        }
        return names, funcs


def _blocks(snap: Snapshot) -> int:
    return sum(sum(row) for row in snap)


def _freeze(occupancy: Sequence[Sequence[int]]) -> Snapshot:
    return tuple(tuple(int(bool(c)) for c in row) for row in occupancy)
