"""Zigbee2MQTT bridge for PirateBot props.

This bridge reads from an existing Z2M MQTT setup without owning it.
Only devices explicitly listed in the Halloween config are mapped to/from
prop mesh topics; all other Z2M traffic is ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from props.broker.mqtt_bridge import MqttBridge, TopicMapping
from props.lib.bus import MessageBus

logger = logging.getLogger("prop_z2m_bridge")


@dataclass
class Z2MDevice:
    friendly_name: str
    mesh_in_topic: Optional[str] = None
    mesh_out_topic: Optional[str] = None
    z2m_set_topic: Optional[str] = None
    payload_field: Optional[str] = None
    value_map: Optional[dict[str, Any]] = None


class Z2MBridge:
    """Opt-in bridge between zigbee2mqtt and the prop mesh."""

    def __init__(
        self,
        bus: MessageBus,
        mqtt_broker_url: str,
        devices: list[Z2MDevice],
        z2m_base: str = "zigbee2mqtt",
    ):
        self.devices = {d.friendly_name: d for d in devices}
        mappings: list[TopicMapping] = []
        for d in devices:
            z2m_state_topic = f"{z2m_base}/{d.friendly_name}"
            if d.mesh_in_topic:
                mappings.append(
                    TopicMapping(
                        mesh_topic=d.mesh_in_topic,
                        mqtt_topic=z2m_state_topic,
                        direction="in",
                        payload_field=d.payload_field,
                    )
                )
            if d.mesh_out_topic and d.z2m_set_topic:
                mappings.append(
                    TopicMapping(
                        mesh_topic=d.mesh_out_topic,
                        mqtt_topic=d.z2m_set_topic,
                        direction="out",
                    )
                )

        def outbound_transform(payload: dict[str, Any], device: Z2MDevice = d) -> Any:
            if device.value_map:
                val = payload.get("state", payload.get("value"))
                return device.value_map.get(str(val), val)
            return payload

        self._outbound_transform = outbound_transform
        self._bridge = MqttBridge(
            bus=bus,
            broker_url=mqtt_broker_url,
            mappings=mappings,
            source_id="z2m_bridge",
        )
        # Monkey-patch MqttBridge payload conversion to handle Z2M value maps.
        self._patch_mqtt_conversion()

    def _patch_mqtt_conversion(self) -> None:
        original_out = self._bridge._to_mqtt_payload

        def _to_z2m_payload(mesh_payload: dict[str, Any], mapping: TopicMapping) -> Any:
            friendly = mapping.mqtt_topic.replace("zigbee2mqtt/", "").replace("/set", "")
            device = self.devices.get(friendly)
            if device and device.value_map:
                val = mesh_payload.get("state", mesh_payload.get("value"))
                mapped = device.value_map.get(str(val), val)
                return {mapping.mqtt_topic.split("/")[-1]: mapped}
            return original_out(mesh_payload, mapping)

        self._bridge._to_mqtt_payload = _to_z2m_payload

    async def start(self) -> None:
        await self._bridge.start()
        logger.info(f"Z2M bridge started with {len(self.devices)} devices")

    async def stop(self) -> None:
        await self._bridge.stop()
