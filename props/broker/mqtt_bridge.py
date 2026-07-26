"""Generic MQTT <-> prop mesh bridge.

Connects to an external MQTT broker and maps configured topic patterns to
and from the mesh message envelope. Does not interpret payloads beyond the
topic-level mapping.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from props.lib.bus import MessageBus
from props.lib.message import Message, Timing

logger = logging.getLogger("prop_mqtt_bridge")


try:
    import aiomqtt
except ImportError:  # pragma: no cover
    aiomqtt = None  # type: ignore


@dataclass
class TopicMapping:
    """Map one MQTT topic to/from one mesh topic."""

    mesh_topic: str
    mqtt_topic: str
    direction: str = "both"  # in | out | both
    qos: int = 0
    payload_field: Optional[str] = None  # extract a single field from MQTT payload


class MqttBridge:
    """Bridge between an MQTT broker and the internal message bus."""

    def __init__(
        self,
        bus: MessageBus,
        broker_url: str,
        mappings: list[TopicMapping],
        source_id: str = "mqtt_bridge",
    ):
        self.bus = bus
        self.broker_url = broker_url
        self.mappings = mappings
        self.source_id = source_id
        self._client: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if aiomqtt is None:
            raise ImportError("aiomqtt is required for MQTT bridge")
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                async with aiomqtt.Client(self.broker_url) as client:
                    self._client = client
                    for mapping in self.mappings:
                        if mapping.direction in ("in", "both"):
                            await client.subscribe(mapping.mqtt_topic, qos=mapping.qos)

                    self.bus.subscribe("*", self._on_mesh_message)

                    async for message in client.messages:
                        mapping = self._find_mapping(message.topic.value)
                        if mapping and mapping.direction in ("in", "both"):
                            try:
                                payload = json.loads(message.payload)
                            except Exception:
                                payload = {"value": message.payload.decode(errors="replace")}
                            mesh_payload = self._to_mesh_payload(payload, mapping)
                            msg = Message(
                                topic=mapping.mesh_topic,
                                source=self.source_id,
                                payload=mesh_payload,
                            )
                            self.bus.publish(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("MQTT bridge error, reconnecting in 5s")
                await asyncio.sleep(5)

    def _on_mesh_message(self, msg: Message) -> None:
        if msg.source == self.source_id:
            return
        for mapping in self.mappings:
            if mapping.direction in ("out", "both") and mapping.mesh_topic == msg.topic:
                payload = self._to_mqtt_payload(msg.payload, mapping)
                if self._client:
                    try:
                        asyncio.create_task(
                            self._client.publish(
                                mapping.mqtt_topic,
                                payload=json.dumps(payload).encode(),
                                qos=mapping.qos,
                            )
                        )
                    except Exception:
                        logger.exception("Failed to publish to MQTT")

    def _find_mapping(self, mqtt_topic: str) -> Optional[TopicMapping]:
        for mapping in self.mappings:
            if mapping.mqtt_topic == mqtt_topic:
                return mapping
        return None

    @staticmethod
    def _to_mesh_payload(mqtt_payload: Any, mapping: TopicMapping) -> dict[str, Any]:
        if isinstance(mqtt_payload, dict):
            if mapping.payload_field:
                return {mapping.payload_field: mqtt_payload.get(mapping.payload_field)}
            return mqtt_payload
        return {"value": mqtt_payload}

    @staticmethod
    def _to_mqtt_payload(mesh_payload: dict[str, Any], mapping: TopicMapping) -> Any:
        if mapping.payload_field:
            return mesh_payload.get(mapping.payload_field)
        return mesh_payload
