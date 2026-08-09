"""
Prop Mesh — event bus for PirateBot and distributed Halloween props.

This module implements the PirateBot side of the prop mesh protocol
defined in props/PROTOCOL.md. It speaks JSON over WebSocket and can
operate as a server (hosting the mesh), a client (connecting to a
broker), or both.

The protocol envelope uses `topic` rather than `type` so it aligns with
the reference broker in props/broker/ and the shared client library in
props/lib/.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _match_topic(pattern: str, topic: str) -> bool:
    """Match a topic against a pattern supporting * and ** globs."""
    if pattern == topic:
        return True
    if pattern == "*":
        return True
    if "**" in pattern or "*" in pattern:
        return fnmatch.fnmatch(topic, pattern)
    return False
@dataclass
class PropEvent:
    """A single event on the prop mesh."""

    topic: str              # e.g. "effects.thunder.clap"
    source: str             # e.g. "piratebot", "fog_machine_01"
    target: Optional[str]   # optional specific target prop; None = broadcast
    payload: dict           # free-form data
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, raw: str) -> Optional["PropEvent"]:
        try:
            data = json.loads(raw)
            return cls(
                topic=data.get("topic", data.get("type", "unknown")),
                source=data.get("source", "unknown"),
                target=data.get("target"),
                payload=data.get("payload", {}),
                timing=data.get("timing", {}),
                meta=data.get("meta", {}),
                timestamp=data.get("timestamp", time.time()),
            )
        except Exception as exc:
            logger.warning(f"Failed to parse prop event: {exc}")
            return None


Handler = Callable[[PropEvent], Any]


class PropMeshBus:
    """
    Event bus that bridges PirateBot to the distributed prop mesh.

    Can run in two modes:
      - server: host a WebSocket server that props connect to.
      - client: connect to an existing mesh broker / another PirateBot.

    Events are broadcast to all connected peers. Local handlers can be
    registered with `on(event_type, handler)`.
    """

    def __init__(
        self,
        source: str = "piratebot",
        mode: str = "server",
        host: str = "0.0.0.0",
        port: int = 9001,
        broker_url: Optional[str] = None,
    ):
        self.source = source
        self.mode = mode
        self.host = host
        self.port = port
        self.broker_url = broker_url

        self._handlers: dict[str, list[Handler]] = {}
        self._clients: set[Any] = set()
        self._server_task: Optional[asyncio.Task] = None
        self._client_task: Optional[asyncio.Task] = None
        self._runner: Optional[Any] = None
        self._running = False

    def on(self, event_type: str, handler: Handler) -> None:
        """Register a local handler for an event type. * matches all."""
        self._handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: Handler) -> None:
        """Remove a local handler."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def emit(
        self,
        event_type: str,
        payload: dict,
        target: Optional[str] = None,
    ) -> None:
        """Emit an event to the mesh and local handlers."""
        event = PropEvent(
            topic=event_type,
            source=self.source,
            target=target,
            payload=payload,
        )
        await self._dispatch_local(event)
        await self._broadcast(event)

    async def _dispatch_local(self, event: PropEvent) -> None:
        """Run registered local handlers. Supports * and prefix.* globs."""
        handlers: list[Handler] = []
        for pattern, h in self._handlers.items():
            if _match_topic(pattern, event.topic):
                handlers.extend(h)
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:
                logger.warning(f"Prop mesh handler error for {event.topic}: {exc}")

    async def _broadcast(self, event: PropEvent) -> None:
        """Send event to all connected WebSocket peers."""
        if not self._clients:
            return

        payload = event.to_json()
        dead: set[Any] = set()
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception as exc:
                logger.warning(f"Failed to send prop event: {exc}")
                dead.add(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def connect(self) -> bool:
        """Start the mesh: server, client, or both depending on config."""
        if self._running:
            return True
        self._running = True

        if self.mode in ("server", "both"):
            await self._start_server()
        if self.mode in ("client", "both"):
            await self._start_client()

        logger.info(f"PropMeshBus running in {self.mode} mode")
        return True

    async def _start_server(self) -> None:
        """Start a WebSocket server that props can connect to."""
        import aiohttp
        from aiohttp import web

        async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self._clients.add(ws)
            logger.info(f"Prop connected to mesh server. Total peers: {len(self._clients)}")
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = PropEvent.from_json(msg.data)
                        if event:
                            logger.debug(f"Received from mesh: {event.topic}")
                            await self._dispatch_local(event)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.warning(f"Prop ws error: {ws.exception()}")
            finally:
                self._clients.discard(ws)
                logger.info(f"Prop disconnected. Total peers: {len(self._clients)}")
            return ws

        app = web.Application()
        app.router.add_get("/ws", websocket_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        self._server_task = asyncio.create_task(site.start())
        self._runner = runner
        logger.info(f"Prop mesh server listening on ws://{self.host}:{self.port}/ws")

    async def _start_client(self) -> None:
        """Connect to an existing mesh broker as a client."""
        if not self.broker_url:
            logger.warning("Prop mesh client mode requested but broker_url is empty")
            return

        async def client_loop() -> None:
            import aiohttp
            while self._running:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(self.broker_url) as ws:
                            logger.info(f"Connected to prop mesh broker: {self.broker_url}")
                            self._clients.add(ws)
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    event = PropEvent.from_json(msg.data)
                                    if event:
                                        await self._dispatch_local(event)
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    break
                            self._clients.discard(ws)
                except Exception as exc:
                    logger.warning(f"Prop mesh broker connection failed: {exc}; retrying in 5s")
                    await asyncio.sleep(5)

        self._client_task = asyncio.create_task(client_loop())

    async def disconnect(self) -> None:
        """Stop the mesh server/client and close all peer connections."""
        self._running = False

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        if self._client_task:
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass

        runner = getattr(self, "_runner", None)
        if runner:
            await runner.cleanup()

        logger.info("PropMeshBus disconnected")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "mode": self.mode,
            "peers": len(self._clients),
            "handlers": {k: len(v) for k, v in self._handlers.items()},
        }
