"""
Interaction State Machine — lightweight per-person state tracking.

Tracks people through their lifecycle (arrive → interact → depart) and
determines the appropriate interaction intent for each event.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InteractionIntent(Enum):
    """What kind of interaction to perform."""

    COSTUME_REACT = "costume_react"  # First-time visitor → full VLM pipeline
    RETURNING = "returning"          # Appearance cache hit → returning greeting
    FAREWELL = "farewell"            # Person departed after interaction


@dataclass
class PersonState:
    """Tracks state for a single tracked person."""

    track_id: int
    first_seen: float = field(default_factory=time.time)
    costume_description: Optional[str] = None
    interacted: bool = False  # Did we complete a full interaction?


class InteractionManager:
    """
    Manages per-person interaction state across track lifecycles.

    Tracks arrivals and departures, and provides state needed to
    determine the correct interaction intent (costume react, returning
    greeting, or farewell).
    """

    def __init__(self):
        self._people: dict[int, PersonState] = {}

    def on_arrival(self, track_id: int) -> PersonState:
        """Register a new person arriving in frame."""
        state = PersonState(track_id=track_id)
        self._people[track_id] = state
        return state

    def on_departure(self, track_id: int) -> Optional[PersonState]:
        """
        Handle a person leaving frame.

        Returns their state (for farewell handling) or None if unknown.
        """
        return self._people.pop(track_id, None)

    def mark_interacted(
        self, track_id: int, description: Optional[str] = None
    ) -> None:
        """Mark that we completed an interaction with this person."""
        state = self._people.get(track_id)
        if state is not None:
            state.interacted = True
            if description is not None:
                state.costume_description = description

    def get_state(self, track_id: int) -> Optional[PersonState]:
        """Get the current state for a tracked person."""
        return self._people.get(track_id)

    @property
    def active_count(self) -> int:
        """Number of people currently being tracked."""
        return len(self._people)
