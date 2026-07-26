#!/usr/bin/env python3
"""
Reference broker for the PirateBot prop mesh.

Supports:
  - WebSocket server that routes JSON messages by topic
  - Optional MQTT bridge
  - UDP multicast discovery replies
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import time
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("prop_mesh_broker")


def _now() -> float:
    return time.time()


class MeshBroker:
    """WebSocket broker with optional MQTT bridge and UDP discovery."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        ws_port: int = 9001,
        udp_port: int = 9002,
        mqtt_url: Optional[str] = None,
        session: str = "halloween-2026",
    ):
        self.host = host
        self.ws_port = ws_port
        self.udp_port = udp_port
        self.mqtt_url = mqtt_url
        self.session = session

        self._peers: dict[str, web.WebSocketResponse] = {}
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        self._running = True
        app = web.Application()
        app.router.add_get("/ws", self._ws_handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=self.host, port=self.ws_port)
        await self._site.start()
        logger.info(f"WebSocket broker listening on ws://{self.host}:{self.ws_port}/ws")

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

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        source: Optional[str] = None
        logger.info(f"Peer connected from {request.remote}")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    source = data.get("source") or source
                    if source and source not in self._peers:
                        self._peers[source] = ws
                    await self._route(data, sender_ws=ws)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"Peer ws error: {ws.exception()}")
        finally:
            if source and source in self._peers:
                del self._peers[source]
            logger.info(f"Peer disconnected: {source}")
        return ws

    async def _route(self, data: dict, sender_ws: web.WebSocketResponse) -> None:
        """Route a message to all subscribers except the sender."""
        target = data.get("target")
        message = json.dumps(data)

        if target:
            peer = self._peers.get(target)
            if peer and peer is not sender_ws:
                try:
                    await peer.send_str(message)
                except Exception as exc:
                    logger.warning(f"Failed to send to {target}: {exc}")
            return

        dead: list[str] = []
        for source_id, peer in list(self._peers.items()):
            if peer is sender_ws:
                continue
            try:
                await peer.send_str(message)
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

            # Join multicast group 239.255.42.99
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


def main() -> int:
    parser = argparse.ArgumentParser(description="PirateBot prop mesh broker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=9001)
    parser.add_argument("--udp-port", type=int, default=9002)
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
        udp_port=args.udp_port,
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
