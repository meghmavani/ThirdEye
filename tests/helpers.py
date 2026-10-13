"""Event constructors shared by the checker tests."""

from __future__ import annotations

from thirdeye.compiler.schema import Step
from thirdeye.events import Event


def placed(step: Step, t: float = 1.0, confidence: float = 1.0, **overrides) -> Event:
    """An ADD event that performs `step` exactly, unless overridden."""
    fields = {"part": step.part, "pose": step.pose, **overrides}
    return Event(t=t, kind="ADD", confidence=confidence, **fields)
