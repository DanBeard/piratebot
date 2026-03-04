"""Tests for the interaction state machine."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.interaction_state import (
    InteractionIntent,
    InteractionManager,
    PersonState,
)


class TestPersonState:
    def test_defaults(self):
        state = PersonState(track_id=1)
        assert state.track_id == 1
        assert state.costume_description is None
        assert state.interacted is False
        assert state.first_seen <= time.time()


class TestInteractionManager:
    def setup_method(self):
        self.mgr = InteractionManager()

    def test_on_arrival_registers_person(self):
        state = self.mgr.on_arrival(42)
        assert state.track_id == 42
        assert self.mgr.active_count == 1

    def test_on_departure_returns_state(self):
        self.mgr.on_arrival(42)
        state = self.mgr.on_departure(42)
        assert state is not None
        assert state.track_id == 42
        assert self.mgr.active_count == 0

    def test_on_departure_unknown_returns_none(self):
        assert self.mgr.on_departure(999) is None

    def test_mark_interacted(self):
        self.mgr.on_arrival(1)
        self.mgr.mark_interacted(1, "skeleton costume")
        state = self.mgr.get_state(1)
        assert state.interacted is True
        assert state.costume_description == "skeleton costume"

    def test_mark_interacted_unknown_track_is_noop(self):
        # Should not raise
        self.mgr.mark_interacted(999, "ghost")

    def test_full_lifecycle(self):
        """arrive → interact → depart → farewell eligible"""
        self.mgr.on_arrival(1)
        assert self.mgr.active_count == 1

        self.mgr.mark_interacted(1, "vampire costume")
        state = self.mgr.get_state(1)
        assert state.interacted is True

        departed = self.mgr.on_departure(1)
        assert departed.interacted is True
        assert departed.costume_description == "vampire costume"
        assert self.mgr.active_count == 0

    def test_departure_without_interaction_not_farewell_eligible(self):
        """Person who left before we interacted should not get farewell."""
        self.mgr.on_arrival(5)
        state = self.mgr.on_departure(5)
        assert state is not None
        assert state.interacted is False

    def test_multiple_people(self):
        self.mgr.on_arrival(1)
        self.mgr.on_arrival(2)
        self.mgr.on_arrival(3)
        assert self.mgr.active_count == 3

        self.mgr.mark_interacted(2, "wizard")
        self.mgr.on_departure(1)
        assert self.mgr.active_count == 2

        state2 = self.mgr.get_state(2)
        assert state2.interacted is True

    def test_get_state_returns_none_for_unknown(self):
        assert self.mgr.get_state(42) is None


class TestInteractionIntent:
    def test_enum_values(self):
        assert InteractionIntent.COSTUME_REACT.value == "costume_react"
        assert InteractionIntent.RETURNING.value == "returning"
        assert InteractionIntent.FAREWELL.value == "farewell"
