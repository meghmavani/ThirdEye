"""The world-state interface (FR-STA-1, NFR-MNT-2).

One interface, two implementations: `state/lego/` keeps a stud grid and
`state/jenga/` keeps a tower lattice. The checker holds a `WorldState` and never
asks which it has, which is the whole of FR-STA-1. Adding a third state model
must require implementing this class and nothing else (NFR-MNT-2), so anything
procedure-specific that leaks in here is a bug in the design, not a shortcut.

A tracker's job is to *observe*, not to simulate. Nothing in the system applies
a spec's declared effects to the state -- the state is whatever the cameras last
confirmed. That is why a hard-dependency violation means our reading is wrong
(FR-CHK-7) rather than that the world is inconsistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from thirdeye.events import Event, Reverification


class WorldState(ABC):
    """What the checker is allowed to know about the physical world."""

    @abstractmethod
    def current(self) -> Any:
        """The latest confirmed state. The type is model-specific by design:
        only the model's own code and its tests read it structurally, while the
        checker sees the world exclusively through `predicates()` and `diff()`."""

    @abstractmethod
    def diff(self) -> list[Event]:
        """Events for everything that changed since the last call, and consume
        them (FR-STA-4, FR-STA-8). Calling twice in a row returns []."""

    def predicates(self) -> tuple[Mapping[str, Any], Mapping[str, Callable[..., Any]]]:
        """The `(names, functions)` this model contributes to precondition
        expressions, drawn from the vocabulary in `compiler/expr.py`.

        A model supplies only what it can actually observe. A stud grid has no
        `top_complete_layer`, so it does not offer one, and a spec that reads it
        is refused before the session starts rather than failing mid-run.
        """
        return {}, {}

    # --- re-verification (FR-STA-6) -------------------------------------------------
    #
    # Four generic methods, because every part of this except "is the view clear
    # yet" is the same for any state model. Note that a tracker never decides
    # *what* was confirmed: mapping a step id to a part and a pose needs the
    # spec, and a tracker that read specs could not be reused for a second
    # procedure without change (NFR-MNT-1). The caller holds the spec and passes
    # the answer in.

    def request_reverify(self, step_ids: set[str]) -> None:
        """Ask for a second, unobstructed look at named prior steps.

        Called by the checker when a hard dependency appears unmet. The request
        is queued rather than served: re-reading through the same occluding hand
        would reproduce the mistake that caused it.
        """
        self._reverify_pending().update(step_ids)
        self._reverify_clear = False

    def pending_reverify(self) -> set[str]:
        return set(self._reverify_pending())

    def reverify_ready(self) -> bool:
        """True once a trustworthy reading has been taken since the request.

        The base answer is "as soon as one was asked for"; a model that can tell
        when its view is obstructed should say so instead.
        """
        return bool(self._reverify_pending()) and getattr(self, "_reverify_clear", False)

    def take_reverification(self, supported: set[str]) -> Reverification | None:
        """Close out the request, or None while the view is still obstructed."""
        if not self.reverify_ready():
            return None
        self._reverify_pending().clear()
        self._reverify_clear = False
        return Reverification(t=getattr(self, "_t", 0.0), supported=set(supported))

    def _reverify_pending(self) -> set[str]:
        if not hasattr(self, "_reverify"):
            self._reverify: set[str] = set()
        return self._reverify
