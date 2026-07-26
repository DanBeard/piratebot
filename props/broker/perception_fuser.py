"""World-state perception fuser and rule engine for the prop mesh.

Fuses raw sensor events and abstract scene inputs into a coherent
world model, then emits derived `world.*` events and triggers effect
topics based on configurable rules.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from props.lib.bus import MessageBus
from props.lib.message import Message, Meta, Timing

logger = logging.getLogger("prop_perception_fuser")


@dataclass
class ZoneState:
    occupied: bool = False
    occupied_since: Optional[float] = None
    last_source: str = ""
    last_seen: float = 0.0
    confidence: float = 0.0


@dataclass
class WorldModel:
    zones: dict[str, ZoneState] = field(default_factory=dict)
    scene: str = "idle"
    quiet_mode: bool = False
    estop: bool = False
    garage_active: bool = False
    garage_active_since: Optional[float] = None


@dataclass
class Rule:
    when: dict[str, Any]
    then: list[dict[str, Any]]
    cooldown_ms: int = 0
    last_fired: float = 0.0


class PerceptionFuser:
    """Maintains a fused world model and runs effect rules."""

    def __init__(
        self,
        bus: MessageBus,
        rules: Optional[list[Rule]] = None,
        zone_timeout_ms: float = 5000,
        source_id: str = "perception",
    ):
        self.bus = bus
        self.world = WorldModel()
        self.rules = rules or []
        self.zone_timeout_ms = zone_timeout_ms
        self.source_id = source_id
        self._seq = 0

        self.bus.subscribe("sensors.*", self._on_sensor)
        self.bus.subscribe("pirate.*", self._on_pirate)
        self.bus.subscribe("scene.*", self._on_scene)
        self.bus.subscribe("world.*", self._on_world)

        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, topic: str, payload: dict[str, Any]) -> None:
        msg = Message(
            topic=topic,
            source=self.source_id,
            payload=payload,
            meta=Meta(seq=self._next_seq()),
            timestamp=time.time(),
        )
        self.bus.publish(msg)

    def _on_sensor(self, msg: Message) -> None:
        topic = msg.topic
        parts = topic.split(".")
        if len(parts) < 3:
            return
        sensor_type = parts[1]
        zone = parts[2]

        payload = msg.payload
        detected = bool(payload.get("detected", payload.get("occupancy", payload.get("pressed", False))))

        state = self.world.zones.setdefault(zone, ZoneState())
        state.last_seen = time.time()
        state.last_source = msg.source
        state.confidence = float(payload.get("confidence", 1.0))

        if detected and not state.occupied:
            state.occupied = True
            state.occupied_since = time.time()
            self._emit(f"world.{zone}.occupied", {"zone": zone, "source": msg.source, "confidence": state.confidence})
            self._try_rules()
        elif not detected and state.occupied:
            state.occupied = False
            state.occupied_since = None
            self._emit(f"world.{zone}.vacant", {"zone": zone, "source": msg.source})

    def _on_pirate(self, msg: Message) -> None:
        # Treat PirateBot arrival/departure as a virtual porch zone.
        if msg.topic == "pirate.arrival":
            self._mark_zone("porch", True, msg.source, msg.payload.get("confidence", 1.0))
        elif msg.topic == "pirate.departure":
            self._mark_zone("porch", False, msg.source)
        self._try_rules()

    def _on_scene(self, msg: Message) -> None:
        if msg.topic == "scene.start":
            self.world.scene = msg.payload.get("scene", "show")
        elif msg.topic == "scene.stop":
            self.world.scene = "idle"
        elif msg.topic == "scene.pause":
            pass
        elif msg.topic == "scene.estop":
            self.world.estop = bool(msg.payload.get("active", True))
            self._emit("world.estop", {"active": self.world.estop})
        elif msg.topic == "scene.resume":
            self.world.estop = False
            self._emit("world.estop", {"active": False})
        self._try_rules()

    def _on_world(self, msg: Message) -> None:
        # Allow external systems to assert world facts (e.g. manual override).
        if msg.topic == "world.garage.start":
            self.world.garage_active = True
            self.world.garage_active_since = time.time()
        elif msg.topic == "world.garage.end":
            self.world.garage_active = False
            self.world.garage_active_since = None
        elif msg.topic == "world.quiet":
            self.world.quiet_mode = bool(msg.payload.get("enabled", True))

    def _mark_zone(self, zone: str, occupied: bool, source: str, confidence: float = 1.0) -> None:
        state = self.world.zones.setdefault(zone, ZoneState())
        state.last_seen = time.time()
        state.last_source = source
        state.confidence = confidence
        if occupied and not state.occupied:
            state.occupied = True
            state.occupied_since = time.time()
            self._emit(f"world.{zone}.occupied", {"zone": zone, "source": source, "confidence": confidence})
        elif not occupied and state.occupied:
            state.occupied = False
            state.occupied_since = None
            self._emit(f"world.{zone}.vacant", {"zone": zone, "source": source})

    async def _cleanup_loop(self) -> None:
        while self._running:
            await asyncio.sleep(1)
            now = time.time()
            for zone, state in list(self.world.zones.items()):
                if state.occupied and (now - state.last_seen) * 1000 > self.zone_timeout_ms:
                    state.occupied = False
                    state.occupied_since = None
                    self._emit(f"world.{zone}.vacant", {"zone": zone, "source": "timeout"})

    def _try_rules(self) -> None:
        if self.world.estop:
            return
        for rule in self.rules:
            if self._matches(rule.when):
                now = time.time()
                if (now - rule.last_fired) * 1000 < rule.cooldown_ms:
                    continue
                rule.last_fired = now
                for action in rule.then:
                    topic = action["topic"]
                    payload = dict(action.get("payload", {}))
                    timing = action.get("timing", {})
                    self._emit(
                        topic,
                        {
                            **payload,
                            "_rule": True,
                        },
                    )

    def _matches(self, when: dict[str, Any]) -> bool:
        for key, expected in when.items():
            actual = self._world_value(key)
            if actual is None:
                return False
            if isinstance(expected, (list, tuple)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _world_value(self, key: str) -> Any:
        if key == "scene":
            return self.world.scene
        if key == "quiet_mode":
            return self.world.quiet_mode
        if key == "estop":
            return self.world.estop
        if key == "garage_active":
            return self.world.garage_active
        if key.startswith("world."):
            key = key[len("world."):]
        if key.endswith(".occupied"):
            zone = key[: -len(".occupied")]
            return self.world.zones.get(zone, ZoneState()).occupied
        if key.endswith(".vacant"):
            zone = key[: -len(".vacant")]
            return not self.world.zones.get(zone, ZoneState()).occupied
        return None
