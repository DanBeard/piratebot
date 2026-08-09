"""Tests for person tracker (real and mock)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from interfaces.detector import Detection, IDetector
from props.lib.bus import MessageBus
from props.lib.message import Message
from services.person_tracker import MockPersonTracker, PersonTracker, ZonePolygon
from services.prop_mesh import PropMeshBus


class FakeDetector(IDetector):
    """Deterministic detector for tests."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detections

    def detect_people(self, frame: np.ndarray) -> list[Detection]:
        return self._detections


@pytest.fixture
def mesh_bus() -> PropMeshBus:
    return PropMeshBus(source="test", mode="client")


@pytest.mark.asyncio
async def test_real_tracker_publishes_zone_update(mesh_bus: PropMeshBus) -> None:
    zones = [
        ZonePolygon("front_yard", 0.0, 0.0, 1.0, 0.5),
        ZonePolygon("driveway", 0.0, 0.5, 1.0, 1.0),
    ]
    det = Detection(x1=10, y1=400, x2=60, y2=700, confidence=0.9, label="person", track_id=1)
    detector = FakeDetector([det])

    received: list[Message] = []
    mesh_bus.on("tracker.person.update", lambda ev: received.append(Message(
        topic=ev.topic, source=ev.source, payload=ev.payload, timestamp=ev.timestamp
    )))
    mesh_bus.on("tracker.person.lost", lambda ev: received.append(Message(
        topic=ev.topic, source=ev.source, payload=ev.payload, timestamp=ev.timestamp
    )))

    tracker = PersonTracker(
        mesh=mesh_bus,
        detector=detector,
        zones=zones,
        capture_callback=lambda: np.zeros((800, 600, 3), dtype=np.uint8),
    )
    await tracker.start()
    await asyncio.sleep(0.5)
    await tracker.stop()

    assert len(received) >= 1
    assert received[0].topic == "tracker.person.update"
    assert received[0].payload["zone"] == "driveway"


@pytest.mark.asyncio
async def test_mock_scenario_emits_enter_and_leave(tmp_path: Path, mesh_bus: PropMeshBus) -> None:
    scenario = tmp_path / "scenario.jsonl"
    scenario.write_text(
        '{"delay_s": 0.05, "action": "enter", "zone": "sideyard", "costume_description": "vampire"}\n'
        '{"delay_s": 0.1, "action": "leave", "zone": "sideyard"}\n'
    )

    received: list[Message] = []
    mesh_bus.on("tracker.person.*", lambda ev: received.append(Message(
        topic=ev.topic, source=ev.source, payload=ev.payload, timestamp=ev.timestamp
    )))

    tracker = MockPersonTracker(mesh=mesh_bus, scenario_path=scenario)
    await tracker.start()
    await asyncio.sleep(0.3)
    await tracker.stop()

    topics = [m.topic for m in received]
    assert "tracker.person.update" in topics
    assert "tracker.person.lost" in topics
    update = next(m for m in received if m.topic == "tracker.person.update")
    assert update.payload["zone"] == "sideyard"
    assert update.payload["costume_description"] == "vampire"
