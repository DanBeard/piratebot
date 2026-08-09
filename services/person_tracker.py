"""Person tracker that publishes `tracker.person.*` events to the prop mesh.

Two modes:
  - real: uses YOLOv8 detections + a zone polygon map to assign people to zones.
  - mock: reads a JSONL scenario or keyboard input to inject fake tracks.

Outputs mesh messages:
  - tracker.person.update {id, zone, confidence, costume_description?}
  - tracker.person.lost {id}
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from interfaces.detector import Detection, IDetector
from services.prop_mesh import PropMeshBus, PropEvent

logger = logging.getLogger(__name__)


@dataclass
class ZonePolygon:
    """Axis-aligned bounding-box zone for simple zone assignment."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class PersonTracker:
    """Assigns detected people to zones and publishes mesh updates."""

    def __init__(
        self,
        mesh: PropMeshBus,
        detector: IDetector,
        zones: list[ZonePolygon],
        capture_callback: Optional[Callable[[], np.ndarray]] = None,
        source: str = "person_tracker",
        loop_interval: float = 0.2,
        lost_timeout_s: float = 5.0,
    ):
        self.mesh = mesh
        self.detector = detector
        self.zones = zones
        self.capture_callback = capture_callback
        self.source = source
        self.loop_interval = loop_interval
        self.lost_timeout_s = lost_timeout_s

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tracks: dict[int, dict[str, Any]] = {}

    def _zone_for(self, x: float, y: float) -> Optional[str]:
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.name
        return None

    async def _publish_update(self, track_id: int, zone: str, confidence: float) -> None:
        event = PropEvent(
            topic="tracker.person.update",
            source=self.source,
            target=None,
            payload={
                "id": f"p{track_id}",
                "zone": zone,
                "confidence": confidence,
            },
        )
        await self.mesh._dispatch_local(event)
        if self.mesh._clients:
            await self.mesh._broadcast(event)

    async def _publish_lost(self, track_id: int) -> None:
        event = PropEvent(
            topic="tracker.person.lost",
            source=self.source,
            target=None,
            payload={"id": f"p{track_id}"},
        )
        await self.mesh._dispatch_local(event)
        if self.mesh._clients:
            await self.mesh._broadcast(event)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Person tracker started (real mode)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Person tracker stopped")

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.loop_interval)
            if self.capture_callback is None:
                continue
            try:
                frame = self.capture_callback()
                if frame is None:
                    continue
                detections = self.detector.detect_people(frame)
                now = time.time()
                active_ids: set[int] = set()

                for det in detections:
                    tid = det.track_id
                    if tid is None:
                        # Fallback: hash box center into a pseudo-id.
                        tid = hash((round(det.center[0], -1), round(det.center[1], -1))) % 100000
                    active_ids.add(tid)
                    zone = self._zone_for(det.center[0] / frame.shape[1], det.center[1] / frame.shape[0])
                    if zone is None:
                        continue
                    self._tracks[tid] = {"zone": zone, "seen": now, "conf": det.confidence}
                    await self._publish_update(tid, zone, det.confidence)

                stale = [tid for tid, meta in self._tracks.items() if tid not in active_ids and now - meta["seen"] > self.lost_timeout_s]
                for tid in stale:
                    await self._publish_lost(tid)
                    self._tracks.pop(tid, None)
            except Exception:
                logger.exception("Person tracker loop failed")


class MockPersonTracker:
    """Inject fake tracker events for testing rules without cameras."""

    def __init__(
        self,
        mesh: PropMeshBus,
        scenario_path: Optional[Path] = None,
        source: str = "mock_tracker",
        keyboard: bool = False,
    ):
        self.mesh = mesh
        self.scenario_path = scenario_path
        self.source = source
        self.keyboard = keyboard

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        event = PropEvent(
            topic=topic,
            source=self.source,
            target=None,
            payload=payload,
        )
        await self.mesh._dispatch_local(event)
        if self.mesh._clients:
            await self.mesh._broadcast(event)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.keyboard:
            self._task = asyncio.create_task(self._keyboard_loop())
        else:
            self._task = asyncio.create_task(self._scenario_loop())
        logger.info("Mock person tracker started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Mock person tracker stopped")

    async def _scenario_loop(self) -> None:
        """Play a JSONL scenario where each line is {"delay_s": 0, "action": "enter", "zone": "driveway"}."""
        if not self.scenario_path or not self.scenario_path.exists():
            logger.warning("No scenario file; mock tracker idling")
            while self._running:
                await asyncio.sleep(1)
            return

        with open(self.scenario_path) as f:
            steps = [json.loads(line) for line in f if line.strip()]

        pid_counter = 0
        active: dict[str, int] = {}
        for step in steps:
            if not self._running:
                break
            await asyncio.sleep(step.get("delay_s", 0))
            action = step.get("action")
            zone = step.get("zone", "unknown")
            costume = step.get("costume_description")

            if action == "enter":
                pid_counter += 1
                pid = f"mock{pid_counter}"
                active[zone] = pid
                await self._emit(
                    "tracker.person.update",
                    {"id": pid, "zone": zone, "confidence": 0.9, "costume_description": costume},
                )
            elif action == "move":
                pid = active.get(step.get("from", zone))
                if pid:
                    active.pop(step.get("from", zone), None)
                    active[zone] = pid
                    await self._emit("tracker.person.update", {"id": pid, "zone": zone, "confidence": 0.9})
            elif action == "leave":
                pid = active.pop(zone, None)
                if pid:
                    await self._emit("tracker.person.lost", {"id": pid})

    async def _keyboard_loop(self) -> None:
        """Read keypresses from stdin and inject tracker events."""
        pid_counter = 0
        zone_map = {
            "1": "sidewalk",
            "2": "front_yard",
            "3": "driveway",
            "4": "graveyard",
            "5": "sideyard",
        }
        active: dict[str, str] = {}
        print("Mock tracker keys: 1=sidewalk 2=front_yard 3=driveway 4=graveyard 5=sideyard x=leave_all q=quit")
        while self._running:
            # Use a thread to avoid blocking the event loop.
            key = await asyncio.get_event_loop().run_in_executor(None, input, "tracker> ")
            key = key.strip().lower()
            if key == "q":
                break
            if key == "x":
                for pid in list(active.values()):
                    await self._emit("tracker.person.lost", {"id": pid})
                active.clear()
                continue
            zone = zone_map.get(key)
            if not zone:
                continue
            if zone in active:
                await self._emit("tracker.person.lost", {"id": active[zone]})
            pid_counter += 1
            pid = f"mock{pid_counter}"
            active[zone] = pid
            await self._emit("tracker.person.update", {"id": pid, "zone": zone, "confidence": 0.9})
