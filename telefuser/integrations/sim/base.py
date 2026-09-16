"""Simulator adapter protocol consumed by the generic VLA runtime."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from telefuser.vla.contracts import RobotObservation
from telefuser.vla.runtime.executor import RobotAction


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Minimal interface implemented by one simulator connection."""

    def observe(self) -> RobotObservation:
        """Capture the latest semantically described robot observation."""
        ...

    def execute(self, action: RobotAction) -> None:
        """Submit one semantically described robot action."""
        ...

    def reset(self) -> RobotObservation:
        """Reset the simulator and return its initial observation."""
        ...
