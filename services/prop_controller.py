"""
PropController — maps PirateBot runtime behavior to prop mesh events.

This sits between the orchestrator, the voice line database, and the
PropMeshBus. It decides when to fire fog, thunder, strobes, or other
effects based on:
  - the emotion tag of a spoken line
  - explicit prop_trigger tags in voice line metadata
  - arrival/departure lifecycle events
  - idle/ambient events

It also listens to incoming mesh events so other props can make the
pirate react, e.g. a motion sensor prop could send `pirate.speak` to
force a specific line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from services.prop_mesh import PropMeshBus, PropEvent

logger = logging.getLogger(__name__)


class PropController:
    """
    High-level bridge between PirateBot and the distributed prop mesh.
    """

    def __init__(
        self,
        mesh: PropMeshBus,
        avatar=None,
        auto_triggers: Optional[dict[str, list[dict]]] = None,
    ):
        self.mesh = mesh
        self.avatar = avatar
        self.auto_triggers = auto_triggers or {}

        # Register incoming handlers
        self.mesh.on("pirate.speak", self._handle_pirate_speak)
        self.mesh.on("pirate.expression", self._handle_pirate_expression)
        self.mesh.on("pirate.animation", self._handle_pirate_animation)
        self.mesh.on("pirate.gaze", self._handle_pirate_gaze)
        self.mesh.on("*", self._log_unknown_event)

    # ------------------------------------------------------------------
    # Outgoing: PirateBot -> props
    # ------------------------------------------------------------------

    async def on_arrival(self, track_id: int) -> None:
        """Someone arrived at the porch."""
        await self.mesh.emit("pirate.arrival", {"track_id": track_id})

    async def on_departure(self, track_id: int) -> None:
        """Someone left."""
        await self.mesh.emit("pirate.departure", {"track_id": track_id})

    async def on_speak(
        self,
        line_id: str,
        text: str,
        emotion: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        """The pirate is about to speak a line. Trigger props accordingly."""
        tags = tags or []
        triggers: list[PropEvent] = []

        # 1. Explicit prop tags on the voice line.
        for tag in tags:
            if tag.startswith("prop:"):
                event_type = tag.split(":", 1)[1]
                triggers.append(
                    PropEvent(
                        type=event_type,
                        source="piratebot",
                        target=None,
                        payload={"line_id": line_id, "text": text, "emotion": emotion},
                    )
                )

        # 2. Emotion-based auto triggers from config.
        for event_type, rules in self.auto_triggers.items():
            for rule in rules:
                if rule.get("emotion") == emotion:
                    triggers.append(
                        PropEvent(
                            type=event_type,
                            source="piratebot",
                            target=None,
                            payload={"line_id": line_id, "text": text, "emotion": emotion},
                        )
                    )
                    break  # only emit once per event_type per line

        for event in triggers:
            logger.info(f"Triggering prop event {event.type} for line {line_id}")
            await self.mesh.emit(event.type, event.payload)

    async def on_idle(self, line_id: str, text: str) -> None:
        """Idle line spoken; low-key ambient prop events only."""
        await self.mesh.emit("pirate.idle_speak", {"line_id": line_id, "text": text})

    # ------------------------------------------------------------------
    # Incoming: props -> PirateBot
    # ------------------------------------------------------------------

    async def _handle_pirate_speak(self, event: PropEvent) -> None:
        """Another prop asked us to say a specific line."""
        if not self.avatar:
            logger.debug("No avatar available for remote pirate.speak")
            return
        audio_url = event.payload.get("audio_url")
        visemes = event.payload.get("visemes")
        emotion = event.payload.get("emotion", "neutral")
        if audio_url:
            await self.avatar.play_audio(audio_url, visemes, emotion)

    async def _handle_pirate_expression(self, event: PropEvent) -> None:
        if not self.avatar:
            return
        from interfaces.avatar_controller import Expression
        try:
            expr = Expression(event.payload.get("expression", "neutral"))
            await self.avatar.set_expression(expr)
        except ValueError:
            logger.warning(f"Unknown expression from mesh: {event.payload}")

    async def _handle_pirate_animation(self, event: PropEvent) -> None:
        if not self.avatar:
            return
        from interfaces.avatar_controller import Animation
        try:
            anim = Animation(event.payload.get("animation", "idle"))
            await self.avatar.play_animation(anim)
        except ValueError:
            logger.warning(f"Unknown animation from mesh: {event.payload}")

    async def _handle_pirate_gaze(self, event: PropEvent) -> None:
        if not self.avatar:
            return
        from interfaces.avatar_controller import GazeTarget
        x = event.payload.get("x", 0.5)
        y = event.payload.get("y", 0.5)
        await self.avatar.set_gaze(GazeTarget(x=x, y=y))

    async def _log_unknown_event(self, event: PropEvent) -> None:
        """Log events that don't have a specific handler."""
        if not event.type.startswith("pirate."):
            logger.debug(f"Prop mesh event received: {event.type} from {event.source}")
