#!/usr/bin/env python3
"""Reference broker for the PirateBot prop mesh.

Serves the WebSocket mesh, UDP discovery, and the control center static
files from a single process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web

from props.lib.bus import MessageBus
from props.lib.message import Message

logger = logging.getLogger("prop_mesh_broker")


def _now() -> float:
    return time.time()


class MeshBroker:
    """WebSocket broker with message bus, optional MQTT bridge, UDP discovery,
    and static file serving for the control center."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        ws_port: int = 9001,
        http_port: int = 9000,
        udp_port: int = 9002,
        static_dir: Optional[Path] = None,
        mqtt_url: Optional[str] = None,
        session: str = "halloween-2026",
    ):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.udp_port = udp_port
        self.static_dir = static_dir
        self.mqtt_url = mqtt_url
        self.session = session

        self.bus = MessageBus()
        self._peers: dict[str, web.WebSocketResponse] = {}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)

        if self.static_dir and self.static_dir.exists():
            app.router.add_static("/assets/", self.static_dir / "assets", name="assets")
            app.router.add_get("/", self._serve_index)
            logger.info(f"Serving control center from {self.static_dir}")
        else:
            logger.warning(
                f"Control center dir not found: {self.static_dir}. "
                "Run `npm run build` in props/control_center."
            )
            app.router.add_get("/", self._index_placeholder)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner, host=self.host, port=self.http_port
        )
        await self._site.start()
        logger.info(
            f"HTTP/control center listening on http://{self.host}:{self.http_port}"
        )

        self._tasks.append(asyncio.create_task(self._udp_discovery_loop()))

        if self.mqtt_url:
            self._tasks.append(asyncio.create_task(self._mqtt_bridge_loop()))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for ws in list(self._peers.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._peers.clear()

        if self._runner:
            await self._runner.cleanup()
        logger.info("Broker stopped")

    async def _index_placeholder(self, _: web.Request) -> web.Response:
        return web.Response(
            text=(
                "PirateBot Prop Mesh Broker\n"
                "WebSocket: /ws\n"
                "Control center build not found."
            ),
            content_type="text/plain",
        )

    async def _serve_index(self, _: web.Request) -> web.Response:
        index_path = self.static_dir / "index.html"
        return web.FileResponse(index_path) if index_path.exists() else self._index_placeholder(_)

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        source: Optional[str] = None
        logger.info(f"Peer connected from {request.remote}")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON from WebSocket")
                        continue
                    source = data.get("source") or source
                    if source and source not in self._peers:
                        self._peers[source] = ws
                    message = Message.from_dict(data)
                    self.bus.publish(message)
                    await self._route(message, sender_ws=ws)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"Peer ws error: {ws.exception()}")
        finally:
            if source and source in self._peers:
                del self._peers[source]
            logger.info(f"Peer disconnected: {source}")
        return ws

    async def _route(self, msg: Message, sender_ws: web.WebSocketResponse) -> None:
        """Route a message to all WebSocket subscribers except the sender."""
        raw = msg.to_json()
        if msg.target:
            peer = self._peers.get(msg.target)
            if peer and peer is not sender_ws:
                try:
                    await peer.send_str(raw)
                except Exception as exc:
                    logger.warning(f"Failed to send to {msg.target}: {exc}")
            return

        dead: list[str] = []
        for source_id, peer in list(self._peers.items()):
            if peer is sender_ws:
                continue
            try:
                await peer.send_str(raw)
            except Exception as exc:
                logger.warning(f"Failed to send to {source_id}: {exc}")
                dead.append(source_id)
        for sid in dead:
            self._peers.pop(sid, None)

    async def _udp_discovery_loop(self) -> None:
        """Listen for UDP multicast discovery requests and reply."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.udp_port))

            mreq = socket.inet_aton("239.255.42.99") + socket.inet_aton("0.0.0.0")
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.setblocking(False)

            loop = asyncio.get_event_loop()
            logger.info(f"UDP discovery listening on {self.udp_port}")

            while self._running:
                try:
                    data, addr = await loop.sock_recvfrom(sock, 1024)
                    try:
                        req = json.loads(data.decode())
                    except Exception:
                        continue
                    if req.get("cmd") == "discover":
                        reply = {
                            "broker": f"ws://{self._get_lan_ip()}:{self.ws_port}/ws",
                            "http": f"http://{self._get_lan_ip()}:{self.http_port}",
                            "session": self.session,
                            "timestamp": _now(),
                        }
                        await loop.sock_sendto(sock, json.dumps(reply).encode(), addr)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning(f"UDP discovery error: {exc}")
                    await asyncio.sleep(1)
        except Exception as exc:
            logger.warning(f"UDP discovery setup failed: {exc}")

    def _get_lan_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    async def _mqtt_bridge_loop(self) -> None:
        """Placeholder for MQTT bridge."""
        logger.warning("MQTT bridge not yet implemented")


class BrokerWithBus:
    """Convenience wrapper that owns a broker and exposes its message bus."""

    def __init__(self, broker: MeshBroker) -> None:
        self.broker = broker
        self.bus = broker.bus

    async def start(self) -> None:
        await self.broker.start()

    async def stop(self) -> None:
        await self.broker.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="PirateBot prop mesh broker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=9001)
    parser.add_argument("--http-port", type=int, default=9000)
    parser.add_argument("--udp-port", type=int, default=9002)
    parser.add_argument("--static-dir", type=str, default="props/control_center/dist")
    parser.add_argument("--mqtt-url")
    parser.add_argument("--session", default="halloween-2026")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    broker = MeshBroker(
        host=args.host,
        ws_port=args.ws_port,
        http_port=args.http_port,
        udp_port=args.udp_port,
        static_dir=Path(args.static_dir).expanduser(),
        mqtt_url=args.mqtt_url,
        session=args.session,
    )

    async def run() -> None:
        await broker.start()
        while True:
            await asyncio.sleep(1)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        asyncio.run(broker.stop())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
