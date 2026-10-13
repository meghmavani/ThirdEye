"""How events are matched to actions, and what an unmatched event means.

The engine is procedure-agnostic (FR-CHK-2), but *matching* cannot be: pairing a
placement with a step needs a part vocabulary and a stud grid, and Jenga has
neither. Rather than branch inside the engine, the state-model-specific part is
isolated behind this protocol and selected from the spec's own `state_model`
field. Adding a state model means adding a policy and a `WorldState`; the engine
is untouched (NFR-MNT-2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from thirdeye.compiler.schema import (
    ConstraintSpec,
    DagSpec,
    JigsawSpec,
    LoweredAction,
    Precondition,
)
from thirdeye.events import Alert, Event, Status


@runtime_checkable
class MatchPolicy(Protocol):
    """The state-model-specific half of judging an event."""

    def match(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> LoweredAction | None:
        """The action this event performs, or None if nothing fits."""

    def unmatched(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """What an event matching no action means. May be nothing."""

    def on_remove(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """What a removal means. May mutate `status`."""

    def matches_hint(self, ev: Event, action: LoweredAction) -> bool:
        """Does a temporal step hint plausibly refer to this action?"""

    def explain(self, pre: Precondition, action: LoweredAction, ev: Event) -> str:
        """Plain-language text for a violated precondition (NFR-USE-2)."""

    def missed_text(self, action: LoweredAction | None, action_id: str) -> str:
        """Plain-language text for a step that was never performed."""


def policy_for(spec: DagSpec | JigsawSpec | ConstraintSpec) -> MatchPolicy:
    """Select a policy from the spec's declared state model.

    Dispatch is on the `state_model` field, not the spec class: the two are
    independent axes -- a future ordered procedure on a tower lattice would
    reuse the lattice policy without a new branch here.
    """
    from thirdeye.checker.rules_jenga import TowerLatticePolicy
    from thirdeye.checker.rules_jigsaw import PieceGridPolicy
    from thirdeye.checker.rules_lego import StudGridPolicy

    policies: dict[str, type] = {
        "stud_grid": StudGridPolicy,
        "piece_grid": PieceGridPolicy,
        "tower_lattice": TowerLatticePolicy,
    }
    cls = policies.get(spec.state_model)
    if cls is None:  # pragma: no cover - the schema constrains state_model
        raise ValueError(f"no match policy for state model {spec.state_model!r}")
    return cls()
