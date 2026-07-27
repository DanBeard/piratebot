"""Tests for the perception fuser."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from props.broker.perception_fuser import PerceptionFuser, Rule
from props.lib.bus import MessageBus
from props.lib.message import Message


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def fuser(bus: MessageBus) -> PerceptionFuser:
    rules = [
        Rule(name="auto_fire_cannon", when={"audience_present": True, "cannon_cooldown": False}, then=[{"topic": "effects.cannon.fire", "payload": {}}], cooldown_ms=100),
        Rule(name="portrait_greet", when={"world.sideyard.occupied": True, "world.sideyard.linger_s": 0}, then=[{"topic": "pirate.speak", "payload": {"category": "greeting"}}], cooldown_ms=100),
        Rule(name="family_mode", when={"family_mode_request": True}, then=[{"topic": "scene.family_mode", "payload": {"active": True}}], cooldown_ms=100),
    ]
    f = PerceptionFuser(bus=bus, rules=rules, location=(34.0, -118.0))
    return f


@pytest.mark.asyncio
async def test_tracker_person_creates_occupancy(bus: MessageBus, fuser: PerceptionFuser) -> None:
    received: list[Message] = []
    bus.subscribe("world.sideyard.occupied", received.append)

    await fuser.start()
    bus.publish(Message(topic="tracker.person.update", source="cam_side", payload={"id": "p1", "zone": "sideyard", "confidence": 0.9}))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].payload["zone"] == "sideyard"
    assert received[0].payload["count"] == 1
    await fuser.stop()


@pytest.mark.asyncio
async def test_audience_present_triggers_cannon(bus: MessageBus, fuser: PerceptionFuser) -> None:
    received: list[Message] = []
    bus.subscribe("effects.cannon.fire", received.append)

    await fuser.start()
    bus.publish(Message(topic="tracker.person.update", source="cam_front", payload={"id": "p2", "zone": "driveway", "confidence": 0.9}))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    await fuser.stop()


@pytest.mark.asyncio
async def test_cannon_cooldown_blocks_refire(bus: MessageBus, fuser: PerceptionFuser) -> None:
    received: list[Message] = []
    bus.subscribe("effects.cannon.fire", received.append)

    await fuser.start()
    fuser.world.cannon_last_fired = time.time()
    bus.publish(Message(topic="tracker.person.update", source="cam_front", payload={"id": "p3", "zone": "driveway", "confidence": 0.9}))
    await asyncio.sleep(0.05)

    assert len(received) == 0
    await fuser.stop()


@pytest.mark.asyncio
async def test_family_mode_toggle(bus: MessageBus, fuser: PerceptionFuser) -> None:
    received: list[Message] = []
    bus.subscribe("scene.family_mode", received.append)

    await fuser.start()
    bus.publish(Message(topic="scene.family_mode.toggle", source="btn_family", payload={}))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert fuser.world.family_mode is True
    await fuser.stop()


@pytest.mark.asyncio
async def test_linger_computes_from_first_seen(bus: MessageBus, fuser: PerceptionFuser) -> None:
    received: list[Message] = []
    bus.subscribe("pirate.speak", received.append)

    await fuser.start()
    bus.publish(Message(topic="tracker.person.update", source="cam_side", payload={"id": "p4", "zone": "sideyard", "confidence": 0.9}))
    await asyncio.sleep(0.05)
    assert "sideyard" in fuser.world.zones
    assert fuser.world.zones["sideyard"].linger_s < 2.0
    await fuser.stop()
