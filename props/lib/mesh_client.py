#!/usr/bin/env python3
"""
Python async client for the PirateBot prop mesh.

Usage:
    client = MeshClient(broker_url="ws://192.168.0.50:9001/ws", source="my_prop")
    await client.connect()
    client.subscribe("effects.thunder.*", on_thunder)
    await client.publish("sensors.pir.motion", {"zone": "porch"})
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import WSMsgType

logger = logging.getLogger("mesh_client")


@dataclass
class MeshMessage:
    """Parsed mesh message."""

    topic: str
    source: str
    target: Optional[str]
    payload: dict
    timing: dict
    meta: dict
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "timing": self.timing,
            "meta": self.meta,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "MeshMessage":
        return cls(
            topic=data.get("topic", "unknown"),
            source=data.get("source", "unknown"),
            target=data.get("target"),
            payload=data.get("payload", {}),
            timing=data.get("timing", {}),
            meta=data.get("meta", {}),
            timestamp=data.get("timestamp", time.time()),
        )


Handler = Callable[[MeshMessage], Any]


class MeshClient:
    """Async WebSocket client for the prop mesh."""

    def __init__(
        self,
        broker_url: str,
        source: str,
        session: str = "halloween-2026",
        reconnect_seconds: float = 5.0,
    ):
        self.broker_url = broker_url
        self.source = source
        self.session = session
        self.reconnect_seconds = reconnect_seconds

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._handlers: list[tuple[str, Handler]] = []
        self._seq = 0
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    def subscribe(self, topic_pattern: str, handler: Handler) -> None:
        """Subscribe to messages matching a glob topic pattern."""
        self._handlers.append((topic_pattern, handler))

    def unsubscribe(self, topic_pattern: str, handler: Handler) -> None:
        self._handlers = [
            (p, h) for p, h in self._handlers if not (p == topic_pattern and h is handler)
        ]

    async def connect(self) -> bool:
        """Connect to the broker and start the receive loop."""
        self._running = True
        self._session = aiohttp.ClientSession()
        await self._connect_loop()
        return True

    async def _connect_loop(self) -> None:
        while self._running:
            try:
                self._ws = await self._session.ws_connect(self.broker_url)
                logger.info(f"Connected to mesh broker: {self.broker_url}")
                await self._receive_loop()
            except Exception as exc:
                logger.warning(f"Mesh broker connection failed: {exc}; retrying in {self.reconnect_seconds}s")
                await asyncio.sleep(self.reconnect_seconds)

    async def _receive_loop(self) -> None:
        async for msg in self._ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message = MeshMessage.from_dict(data)
                    await self._dispatch(message)
                except Exception as exc:
                    logger.warning(f"Failed to handle mesh message: {exc}")
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                break

    async def _dispatch(self, message: MeshMessage) -> None:
        """Run matching handlers."""
        for pattern, handler in self._handlers:
            if fnmatch.fnmatch(message.topic, pattern):
                try:
                    result = handler(message)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as exc:
                    logger.warning(f"Mesh handler error for {message.topic}: {exc}")

    async def publish(
        self,
        topic: str,
        payload: dict,
        target: Optional[str] = None,
        timing: Optional[dict] = None,
    ) -> None:
        """Publish a message to the mesh."""
        self._seq += 1
        message = MeshMessage(
            topic=topic,
            source=self.source,
            target=target,
            payload=payload,
            timing=timing or {},
            meta={"seq": self._seq, "session": self.session},
            timestamp=time.time(),
        )
        if self._ws and not self._ws.closed:
            await self._ws.send_str(message.to_json())
        else:
            logger.debug(f"Mesh not connected; dropping message {topic}")

    async def disconnect(self) -> None:
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


async def discover_broker(
    udp_port: int = 9002,
    multicast_group: str = "239.255.42.99",
    timeout_seconds: float = 2.0,
) -> Optional[str]:
    """Send a UDP multicast discover request and return the broker URL."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_seconds)

    req = json.dumps({"cmd": "discover", "session": "halloween-2026"}).encode()
    sock.sendto(req, (multicast_group, udp_port))

    try:
        data, _ = sock.recvfrom(1024)
        reply = json.loads(data.decode())
        return reply.get("broker")
    except Exception:
        return None
