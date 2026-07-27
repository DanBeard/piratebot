"""World-state perception fuser and rule engine for the prop mesh.

Fuses person tracks, sensor events, scene commands, and director/mic cues
into a coherent world model, then emits derived `world.*` events and triggers
effect topics based on configurable rules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from props.lib.bus import MessageBus
from props.lib.message import Message, Meta, Timing

logger = logging.getLogger("prop_perception_fuser")


@dataclass
class Person:
    """A tracked person in the world."""

    id: str
    zone: str
    first_seen: float
    last_seen: float
    confidence: float = 1.0
    costume_description: Optional[str] = None

    @property
    def linger_s(self) -> float:
        return time.time() - self.first_seen


@dataclass
class ZoneState:
    """Aggregated state of a physical zone."""

    last_source: str = ""
    people: dict[str, Person] = field(default_factory=dict)

    @property
    def occupied(self) -> bool:
        return bool(self.people)

    @property
    def count(self) -> int:
        return len(self.people)

    @property
    def linger_s(self) -> float:
        if not self.people:
            return 0.0
        return max(p.linger_s for p in self.people.values())


@dataclass
class WorldModel:
    """Fused world state used by rules."""

    # People and zones
    people: dict[str, Person] = field(default_factory=dict)
    zones: dict[str, ZoneState] = field(default_factory=dict)

    # Scene / mode
    scene: str = "idle"
    last_scene: str = "idle"
    family_mode: bool = False
    family_mode_request: Optional[bool] = None

    # Audience
    audience_present: bool = False
    audience_zones: set[str] = field(default_factory=set)

    # Audio/show state
    pumpkins_singing: bool = False
    last_pumpkins_singing: bool = False
    portrait_speaking: bool = False
    costume_description: Optional[str] = None

    # Cooldowns (seconds since last fire; cooldown booleans derived on read)
    fog_last_fired: float = 0.0
    thunder_last_fired: float = 0.0
    cannon_last_fired: float = 0.0

    # Schedule
    sun_down: bool = False
    late_night: bool = False

    # Misc
    quiet_mode: bool = False
    estop: bool = False
    garage_active: bool = False
    garage_active_since: Optional[float] = None

    # Track previous values for edge detection
    last_zone_occupied: dict[str, bool] = field(default_factory=dict)
    last_audience_present: bool = False


@dataclass
class Rule:
    """A single when/then rule with cooldown."""

    name: str
    when: dict[str, Any]
    then: list[dict[str, Any]]
    cooldown_ms: int = 0
    last_fired: float = 0.0


class PerceptionFuser:
    """Maintains a fused world model and runs effect rules."""

    AUDIENCE_ZONES = {"front_yard", "driveway", "sideyard"}
    TRACK_TIMEOUT_S = 15.0
    FOG_COOLDOWN_S = 30.0
    THUNDER_COOLDOWN_S = 45.0
    CANNON_COOLDOWN_S = 30.0

    def __init__(
        self,
        bus: MessageBus,
        rules: Optional[list[Rule]] = None,
        source_id: str = "perception",
        location: tuple[float, float] = (34.0522, -118.2437),
    ):
        self.bus = bus
        self.world = WorldModel()
        self.rules = rules or []
        self.source_id = source_id
        self.location = location
        self._seq = 0

        self.bus.subscribe("sensors.*", self._on_sensor)
        self.bus.subscribe("tracker.*", self._on_tracker)
        self.bus.subscribe("pirate.*", self._on_pirate)
        self.bus.subscribe("scene.*", self._on_scene)
        self.bus.subscribe("world.*", self._on_world)
        self.bus.subscribe("director.*", self._on_director)
        self.bus.subscribe("audio.*", self._on_audio)
        self.bus.subscribe("effects.cannon.*", self._on_cannon_effect)
        self.bus.subscribe("effects.thunder.*", self._on_thunder_effect)
        self.bus.subscribe("effects.fog.*", self._on_fog_effect)

        self._cleanup_task: Optional[asyncio.Task] = None
        self._schedule_task: Optional[asyncio.Task] = None
        self._running = False

    @classmethod
    def from_yaml(cls, bus: MessageBus, path: Path, **kwargs) -> "PerceptionFuser":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        rules = [Rule(name=r["name"], when=r["when"], then=r["then"], cooldown_ms=r.get("cooldown_ms", 0)) for r in data.get("rules", [])]
        return cls(bus, rules=rules, **kwargs)

    async def start(self) -> None:
        self._running = True
        self._update_schedule_state()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._schedule_task = asyncio.create_task(self._schedule_loop())

    async def stop(self) -> None:
        self._running = False
        for task in (self._cleanup_task, self._schedule_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, topic: str, payload: dict[str, Any], timing: Optional[Timing] = None) -> None:
        msg = Message(
            topic=topic,
            source=self.source_id,
            payload=payload,
            meta=Meta(seq=self._next_seq()),
            timestamp=time.time(),
            timing=timing or Timing(),
        )
        self.bus.publish(msg)

    # -------------------------------------------------------------------------
    # Input handlers
    # -------------------------------------------------------------------------

    def _on_sensor(self, msg: Message) -> None:
        """Handle raw Z2M/binary sensor events."""
        parts = msg.topic.split(".")
        if len(parts) < 3:
            return
        zone = parts[2]
        payload = msg.payload
        detected = bool(payload.get("detected", payload.get("occupancy", payload.get("pressed", False))))

        # Treat a sensor detection as an anonymous person in the zone.
        if detected:
            sensor_id = f"sensor_{msg.source}_{zone}"
            now = time.time()
            person = self.world.people.get(sensor_id)
            if person is None:
                person = Person(id=sensor_id, zone=zone, first_seen=now, last_seen=now, confidence=0.7)
                self.world.people[sensor_id] = person
            else:
                person.zone = zone
                person.last_seen = now
                person.confidence = max(person.confidence, float(payload.get("confidence", 0.7)))
        self._reconcile()

    def _on_tracker(self, msg: Message) -> None:
        """Handle camera person-track updates."""
        if msg.topic == "tracker.person.update":
            self._update_person_from_payload(msg.payload)
        elif msg.topic == "tracker.person.lost":
            person_id = msg.payload.get("id")
            if person_id and person_id in self.world.people:
                del self.world.people[person_id]
        self._reconcile()

    def _update_person_from_payload(self, payload: dict[str, Any]) -> None:
        person_id = payload.get("id")
        zone = payload.get("zone")
        if not person_id or not zone:
            return
        now = time.time()
        person = self.world.people.get(person_id)
        if person is None:
            person = Person(
                id=person_id,
                zone=zone,
                first_seen=now,
                last_seen=now,
                confidence=float(payload.get("confidence", 1.0)),
                costume_description=payload.get("costume_description"),
            )
            self.world.people[person_id] = person
        else:
            person.zone = zone
            person.last_seen = now
            person.confidence = float(payload.get("confidence", person.confidence))
            if payload.get("costume_description"):
                person.costume_description = payload["costume_description"]

    def _on_pirate(self, msg: Message) -> None:
        """Handle PirateBot / portrait state updates."""
        if msg.topic == "pirate.speaking":
            self.world.portrait_speaking = bool(msg.payload.get("active", True))
        elif msg.topic == "pirate.finished":
            self.world.portrait_speaking = False
        elif msg.topic == "pirate.costume_description":
            self.world.costume_description = msg.payload.get("description")
        self._try_rules()

    def _on_scene(self, msg: Message) -> None:
        self.world.last_scene = self.world.scene
        if msg.topic == "scene.start":
            self.world.scene = msg.payload.get("scene", "showtime")
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
        elif msg.topic == "scene.family_mode":
            self.world.family_mode = True
            self.world.family_mode_request = True
        elif msg.topic == "scene.spooky_mode":
            self.world.family_mode = False
            self.world.family_mode_request = False
        elif msg.topic == "scene.family_mode.toggle":
            self.world.family_mode = not self.world.family_mode
            self.world.family_mode_request = self.world.family_mode
        self._reconcile()

    def _on_world(self, msg: Message) -> None:
        if msg.topic == "world.garage.start":
            self.world.garage_active = True
            self.world.garage_active_since = time.time()
        elif msg.topic == "world.garage.end":
            self.world.garage_active = False
            self.world.garage_active_since = None
        elif msg.topic == "world.quiet":
            self.world.quiet_mode = bool(msg.payload.get("enabled", True))
        self._try_rules()

    def _on_director(self, msg: Message) -> None:
        if msg.topic == "director.cannon.fire":
            self._try_rules_with_override("manual_cannon", True)
        elif msg.topic == "director.pirate_button":
            self._try_rules_with_override("manual_pirate_button", True)
        elif msg.topic == "director.mic.cue":
            cue = msg.payload.get("cue")
            if cue:
                self._try_rules_with_override("mic_cue", cue)

    def _on_audio(self, msg: Message) -> None:
        if msg.topic == "audio.pumpkin.sing":
            self.world.last_pumpkins_singing = self.world.pumpkins_singing
            self.world.pumpkins_singing = True
            self._reconcile()
        elif msg.topic == "audio.pumpkin.idle":
            self.world.last_pumpkins_singing = self.world.pumpkins_singing
            self.world.pumpkins_singing = False
            self._reconcile()

    def _on_cannon_effect(self, msg: Message) -> None:
        if msg.topic == "effects.cannon.fire":
            self.world.cannon_last_fired = time.time()
            self._emit("world.cannon.fired", {"source": msg.source})
            self._try_rules()

    def _on_thunder_effect(self, msg: Message) -> None:
        if msg.topic == "effects.thunder.clap":
            self.world.thunder_last_fired = time.time()
            self._try_rules()

    def _on_fog_effect(self, msg: Message) -> None:
        if msg.topic == "effects.fog.pulse":
            self.world.fog_last_fired = time.time()
            self._try_rules()

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    def _reconcile(self) -> None:
        """Recompute zone occupancy, audience, and edge states, then run rules."""
        now = time.time()

        # Rebuild zones from people.
        new_zones: dict[str, ZoneState] = {}
        for person in self.world.people.values():
            zone = new_zones.setdefault(person.zone, ZoneState())
            zone.people[person.id] = person

        # Detect goodbye / entrance edges.
        for zone_name, zone in new_zones.items():
            was_occupied = self.world.last_zone_occupied.get(zone_name, False)
            is_occupied = zone.occupied
            if is_occupied and not was_occupied:
                self._emit(f"world.{zone_name}.occupied", {"zone": zone_name, "count": zone.count})
            elif not is_occupied and was_occupied:
                self._emit(f"world.{zone_name}.vacant", {"zone": zone_name})
        for zone_name, was_occupied in self.world.last_zone_occupied.items():
            if zone_name not in new_zones and was_occupied:
                self._emit(f"world.{zone_name}.vacant", {"zone": zone_name})

        self.world.zones = new_zones
        self.world.last_zone_occupied = {z: zs.occupied for z, zs in new_zones.items()}

        # Audience present: driveway / front_yard / sideyard (exclude sidewalk).
        audience_zones = {z for z in self.AUDIENCE_ZONES if z in new_zones and new_zones[z].occupied}
        self.world.audience_present = bool(audience_zones)
        self.world.audience_zones = audience_zones

        if self.world.audience_present != self.world.last_audience_present:
            self._emit(
                "world.audience.present" if self.world.audience_present else "world.audience.absent",
                {"zones": sorted(audience_zones)},
            )
            self.world.last_audience_present = self.world.audience_present

        # Inherit latest costume description from sideyard people.
        sideyard = new_zones.get("sideyard")
        if sideyard:
            for person in sorted(sideyard.people.values(), key=lambda p: p.last_seen, reverse=True):
                if person.costume_description:
                    self.world.costume_description = person.costume_description
                    break

        self._try_rules()

    def _try_rules_with_override(self, key: str, value: Any) -> None:
        """Run rules with a temporary override key set."""
        original = getattr(self.world, key, None)
        setattr(self.world, key, value)
        self._try_rules()
        setattr(self.world, key, original)

    # -------------------------------------------------------------------------
    # Rule engine
    # -------------------------------------------------------------------------

    def _try_rules(self) -> None:
        if self.world.estop:
            return
        now = time.time()
        for rule in self.rules:
            if self._matches(rule.when):
                if (now - rule.last_fired) * 1000 < rule.cooldown_ms:
                    continue
                rule.last_fired = now
                for action in rule.then:
                    topic = action["topic"]
                    payload = dict(action.get("payload", {}))
                    timing = Timing.from_dict(action.get("timing", {}))
                    self._emit(topic, {**payload, "_rule": rule.name}, timing=timing)

    def _matches(self, when: dict[str, Any]) -> bool:
        for key, expected in when.items():
            actual = self._world_value(key)
            if actual is None:
                return False
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _world_value(self, key: str) -> Any:
        if key == "scene":
            return self.world.scene
        if key == "last_scene":
            return self.world.last_scene
        if key == "family_mode":
            return self.world.family_mode
        if key == "family_mode_request":
            return self.world.family_mode_request
        if key == "audience_present":
            return self.world.audience_present
        if key == "pumpkins_singing":
            return self.world.pumpkins_singing
        if key == "last_pumpkins_singing":
            return self.world.last_pumpkins_singing
        if key == "portrait_speaking":
            return self.world.portrait_speaking
        if key == "costume_description":
            return "present" if self.world.costume_description else None
        if key == "sun_down":
            return self.world.sun_down
        if key == "late_night":
            return self.world.late_night
        if key == "quiet_mode":
            return self.world.quiet_mode
        if key == "estop":
            return self.world.estop
        if key == "garage_active":
            return self.world.garage_active
        if key == "manual_cannon":
            return False
        if key == "manual_pirate_button":
            return False
        if key == "mic_cue":
            return None
        if key == "fog_cooldown":
            return (time.time() - self.world.fog_last_fired) < self.FOG_COOLDOWN_S
        if key == "thunder_cooldown":
            return (time.time() - self.world.thunder_last_fired) < self.THUNDER_COOLDOWN_S
        if key == "cannon_cooldown":
            return (time.time() - self.world.cannon_last_fired) < self.CANNON_COOLDOWN_S
        if key == "last_sideyard_occupied":
            return self.world.last_zone_occupied.get("sideyard", False)
        if key.startswith("world."):
            inner = key[len("world."):]
            return self._zone_value(inner)
        return None

    def _zone_value(self, inner: str) -> Any:
        if inner.endswith(".occupied"):
            zone = inner[: -len(".occupied")]
            return self.world.zones.get(zone, ZoneState()).occupied
        if inner.endswith(".linger_s"):
            zone = inner[: -len(".linger_s")]
            return self.world.zones.get(zone, ZoneState()).linger_s
        if inner.endswith(".count"):
            zone = inner[: -len(".count")]
            return self.world.zones.get(zone, ZoneState()).count
        return None

    # -------------------------------------------------------------------------
    # Cleanup & schedule loops
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        while self._running:
            await asyncio.sleep(1)
            now = time.time()
            stale = [pid for pid, p in self.world.people.items() if now - p.last_seen > self.TRACK_TIMEOUT_S]
            if stale:
                for pid in stale:
                    del self.world.people[pid]
                self._reconcile()

    async def _schedule_loop(self) -> None:
        while self._running:
            self._update_schedule_state()
            await asyncio.sleep(60)

    def _update_schedule_state(self) -> None:
        now = time.time()
        was_sun_down = self.world.sun_down
        was_late_night = self.world.late_night
        self.world.sun_down = self._is_sun_down(now)
        self.world.late_night = self._is_late_night(now)
        if self.world.sun_down != was_sun_down:
            self._emit("world.sun", {"down": self.world.sun_down})
        if self.world.late_night != was_late_night:
            self._emit("world.late_night", {"active": self.world.late_night})
        self._try_rules()

    def _is_sun_down(self, now: float) -> bool:
        # Approximate sunset using a simple hour-of-day model for October at the
        # configured latitude. For Los Angeles-ish latitudes, sunset is roughly
        # 18:30 local time in October. Refinements welcome.
        lt = time.localtime(now)
        hour = lt.tm_hour + lt.tm_min / 60.0
        return hour >= 18.5

    def _is_late_night(self, now: float) -> bool:
        lt = time.localtime(now)
        hour = lt.tm_hour + lt.tm_min / 60.0
        return hour >= 23.0
