"""Tests for the MQTT bridge using a mock MQTT broker."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from props.broker.mqtt_bridge import MqttBridge, TopicMapping
from props.lib.bus import MessageBus
from props.lib.message import Message


class FakeMqttClient:
    """In-memory MQTT client that routes between the bridge and a simulated broker."""

    def __init__(self, broker: "FakeMqttBroker") -> None:
        self._broker = broker
        self._bridge: Any = None
        self.subscriptions: list[tuple[str, int]] = []
        self._messages: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        self.subscriptions.append((topic, qos))

    async def publish(self, topic: str, payload: bytes, qos: int = 0) -> None:
        await self._broker.publish(topic, payload, qos, self)

    async def _deliver_remote(self, topic: str, payload: bytes) -> None:
        """Used by the fake broker to inject messages into this client."""
        self.deliver(topic, payload)

    async def _deliver_remote(self, topic: str, payload: bytes) -> None:
        """Used by the fake broker to inject messages into this client."""
        self.deliver(topic, payload)

    def deliver(self, topic: str, payload: bytes) -> None:
        # aiomqtt message shape used by the bridge.
        class _Topic:
            value = topic

        class _Message:
            topic = _Topic()
            payload = b""

        msg = _Message()
        msg.payload = payload
        self._messages.put_nowait(msg)

    @property
    def messages(self):
        class MessagesIterator:
            def __init__(self, client):
                self._client = client

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self._client._messages.get()

        return MessagesIterator(self)


class FakeMqttListener:
    """A second fake client for observing broker-published messages."""

    def __init__(self, broker: FakeMqttBroker) -> None:
        self._client = broker.connect()

    async def subscribe(self, topic: str, qos: int = 0) -> None:
        await self._client.subscribe(topic, qos)

    @property
    def messages(self):
        return self._client.messages


class FakeMqttBroker:
    """Simulated MQTT broker that fans out messages to connected clients."""

    def __init__(self) -> None:
        self.clients: list[FakeMqttClient] = []

    def connect(self) -> FakeMqttClient:
        client = FakeMqttClient(self)
        self.clients.append(client)
        return client

    async def publish(self, topic: str, payload: bytes, qos: int, sender: FakeMqttClient) -> None:
        for client in self.clients:
            if client is sender:
                continue
            for sub_topic, _ in client.subscriptions:
                if _match_wildcard(sub_topic, topic):
                    client.deliver(topic, payload)
                    break


def _match_wildcard(pattern: str, topic: str) -> bool:
    p_parts = pattern.split("/")
    t_parts = topic.split("/")
    if len(p_parts) == 0:
        return False
    for i, part in enumerate(p_parts):
        if part == "#":
            return True
        if part == "+":
            continue
        if i >= len(t_parts) or part != t_parts[i]:
            return False
    return len(p_parts) == len(t_parts)


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def fake_broker() -> FakeMqttBroker:
    return FakeMqttBroker()


@pytest.fixture
def mock_aiomqtt(monkeypatch: Any, fake_broker: FakeMqttBroker) -> FakeMqttClient:
    """Patch aiomqtt.Client so it returns our fake broker client."""
    import props.broker.mqtt_bridge as mb

    real_aiomqtt = mb.aiomqtt
    assert real_aiomqtt is not None, "aiomqtt not installed"

    bridge_client = fake_broker.connect()

    class FakeClientClass:
        _shared = bridge_client

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self._shared

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(mb.aiomqtt, "Client", FakeClientClass)
    return bridge_client


@pytest.mark.asyncio
async def test_mqtt_to_mesh(bus: MessageBus, mock_aiomqtt: FakeMqttClient) -> None:
    received: list[Message] = []
    bus.subscribe("sensors.porch.motion", received.append)

    bridge = MqttBridge(
        bus=bus,
        broker_url="mqtt://fake",
        mappings=[
            TopicMapping(
                mesh_topic="sensors.porch.motion",
                mqtt_topic="zigbee2mqtt/porch_pir",
                direction="in",
            )
        ],
    )
    await bridge.start()
    await asyncio.sleep(0.05)

    mock_aiomqtt.deliver("zigbee2mqtt/porch_pir", json.dumps({"occupancy": True}).encode())
    await asyncio.sleep(0.05)

    await bridge.stop()

    assert len(received) == 1
    assert received[0].topic == "sensors.porch.motion"
    assert received[0].payload["occupancy"] is True


@pytest.mark.asyncio
async def test_mesh_to_mqtt(bus: MessageBus, fake_broker: FakeMqttBroker, mock_aiomqtt: FakeMqttClient) -> None:
    listener = FakeMqttListener(fake_broker)
    await listener.subscribe("props/cannon/fire", 0)

    bridge = MqttBridge(
        bus=bus,
        broker_url="mqtt://fake",
        mappings=[
            TopicMapping(
                mesh_topic="effects.cannon.fire",
                mqtt_topic="props/cannon/fire",
                direction="out",
            )
        ],
    )
    await bridge.start()
    await asyncio.sleep(0.05)

    bus.publish(Message(topic="effects.cannon.fire", source="broker", payload={"delay_ms": 100}))
    await asyncio.sleep(0.1)

    await bridge.stop()

    found = False
    try:
        while True:
            msg = await asyncio.wait_for(listener.messages.__anext__(), timeout=0.5)
            if msg.topic.value == "props/cannon/fire":
                payload = json.loads(msg.payload)
                if payload["delay_ms"] == 100:
                    found = True
                    break
    except asyncio.TimeoutError:
        pass
    assert found


@pytest.mark.asyncio
async def test_bridge_does_not_loopback(bus: MessageBus, fake_broker: FakeMqttBroker, mock_aiomqtt: FakeMqttClient) -> None:
    bridge = MqttBridge(
        bus=bus,
        broker_url="mqtt://fake",
        mappings=[
            TopicMapping(
                mesh_topic="effects.cannon.fire",
                mqtt_topic="props/cannon/fire",
                direction="both",
            )
        ],
    )
    await bridge.start()
    await asyncio.sleep(0.05)

    # Simulate MQTT message arriving from a remote publisher.
    mock_aiomqtt.deliver("props/cannon/fire", json.dumps({"delay_ms": 50}).encode())
    await asyncio.sleep(0.05)

    await bridge.stop()

    # The bridge should have subscribed and processed the inbound message.
    assert ("props/cannon/fire", 0) in mock_aiomqtt.subscriptions
