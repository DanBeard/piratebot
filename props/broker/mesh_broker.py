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

from props.broker.mqtt_bridge import MqttBridge, TopicMapping
from props.broker.perception_fuser import PerceptionFuser
from props.broker.z2m_bridge import Z2MBridge, Z2MDevice
from props.lib.bus import MessageBus
from props.lib.message import Message, Timing
from services.person_tracker import MockPersonTracker, PersonTracker, ZonePolygon
from services.prop_mesh import PropMeshBus
from services.yolo_detector import YoloDetector

logger = logging.getLogger("prop_mesh_broker")


def _now() -> float:
    return time.time()


class MeshBroker:
    """WebSocket broker with message bus, optional MQTT bridge, UDP discovery,
    and static file serving for the control center and displays."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        ws_port: int = 9001,
        http_port: int = 9000,
        udp_port: int = 9002,
        static_dir: Optional[Path] = None,
        display_dirs: Optional[dict[str, Path]] = None,
        mqtt_url: Optional[str] = None,
        session: str = "halloween-2026",
        mock_tracker: Optional[dict[str, Any]] = None,
    ):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self.udp_port = udp_port
        self.static_dir = static_dir
        self.display_dirs = display_dirs or {}
        self.mqtt_url = mqtt_url
        self.session = session
        self.mock_tracker = mock_tracker

        self.mqtt_bridge: Optional[MqttBridge] = None
        self.z2m_bridge: Optional[Z2MBridge] = None
        self.fuser: Optional[PerceptionFuser] = None
        self.tracker: Optional[PersonTracker | MockPersonTracker] = None

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

        for name, path in self.display_dirs.items():
            if not path.exists():
                logger.warning(f"Display dir not found: {path}. Run `npm run build` in props/displays/{name}.")
                continue
            route = f"/{name}/"
            app.router.add_get(route, self._display_index_factory(path))
            app.router.add_static(f"/{name}/assets/", path / "assets", name=f"{name}_assets")
            logger.info(f"Serving display '{name}' from {path} at /{name}/")

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

        # Start an optional built-in mock tracker so the show can be tested
        # without cameras. In production this block is skipped.
        if self.mock_tracker:
            mesh = PropMeshBus(source="broker_tracker", mode="server", host=self.host, port=self.ws_port)
            await mesh.connect()
            kind = self.mock_tracker.get("kind", "scenario")
            if kind == "keyboard":
                self.tracker = MockPersonTracker(mesh=mesh, keyboard=True, source="mock_tracker")
            else:
                scenario = Path(self.mock_tracker.get("scenario", "scenarios/mock_arrival.jsonl"))
                self.tracker = MockPersonTracker(mesh=mesh, scenario_path=scenario, source="mock_tracker")
            await self.tracker.start()
            logger.info(f"Built-in mock tracker started ({kind})")

        if self.mqtt_url:
            await self._start_mqtt_bridges()
        fuser_path = Path(__file__).parent / "fuser_rules.yaml"
        if fuser_path.exists():
            self.fuser = PerceptionFuser.from_yaml(
                bus=self.bus,
                path=fuser_path,
                source_id="perception",
            )
            await self.fuser.start()
            logger.info(f"Perception fuser loaded from {fuser_path}")
        else:
            logger.warning(f"Fuser rules not found at {fuser_path}; fuser disabled")

    async def _start_mqtt_bridges(self) -> None:
        """Start generic MQTT bridge and optional Z2M bridge."""
        generic_mappings = [
            TopicMapping(
                mesh_topic="scene.estop",
                mqtt_topic="piratebot/scene/estop",
                direction="both",
            ),
            TopicMapping(
                mesh_topic="scene.resume",
                mqtt_topic="piratebot/scene/resume",
                direction="both",
            ),
        ]
        self.mqtt_bridge = MqttBridge(
            bus=self.bus,
            broker_url=self.mqtt_url,
            mappings=generic_mappings,
            source_id="mqtt_bridge",
        )
        await self.mqtt_bridge.start()
        logger.info(f"MQTT bridge connected to {self.mqtt_url}")

        # Opt-in Z2M devices. Edit or replace with a config file as hardware is known.
        z2m_devices: list[Z2MDevice] = [
            # Example:
            # Z2MDevice(
            #     friendly_name="porch_pir",
            #     mesh_in_topic="sensors.porch.motion",
            #     payload_field="occupancy",
            # ),
        ]
        if z2m_devices:
            self.z2m_bridge = Z2MBridge(
                bus=self.bus,
                mqtt_broker_url=self.mqtt_url,
                devices=z2m_devices,
            )
            await self.z2m_bridge.start()

    async def stop(self) -> None:
        self._running = False
        if self.tracker:
            await self.tracker.stop()
        if self.fuser:
            await self.fuser.stop()
        if self.mqtt_bridge:
            await self.mqtt_bridge.stop()
        if self.z2m_bridge:
            await self.z2m_bridge.stop()
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

    def _display_index_factory(self, display_dir: Path):
        async def _display_index(_: web.Request) -> web.Response:
            index_path = display_dir / "index.html"
            if not index_path.exists():
                return web.Response(text=f"Display index not found at {index_path}", status=404)
            return web.FileResponse(index_path)
        return _display_index

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
        """Deprecated placeholder; replaced by _start_mqtt_bridges."""
        logger.warning("MQTT bridge loop placeholder is no longer used")


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
    parser.add_argument("--garage-dir", type=str, default="props/displays/garage_ship/dist")
    parser.add_argument("--pumpkins-dir", type=str, default="props/displays/pumpkins/dist")
    parser.add_argument("--mqtt-url")
    parser.add_argument("--session", default="halloween-2026")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--mock-tracker",
        choices=["keyboard", "scenario"],
        help="Run a built-in fake person tracker for testing rules without cameras.",
    )
    parser.add_argument("--mock-scenario", default="scenarios/mock_arrival.jsonl", help="Scenario file for --mock-tracker=scenario")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    mock_tracker = None
    if args.mock_tracker:
        mock_tracker = {"kind": args.mock_tracker, "scenario": args.mock_scenario}

    broker = MeshBroker(
        host=args.host,
        ws_port=args.ws_port,
        http_port=args.http_port,
        udp_port=args.udp_port,
        static_dir=Path(args.static_dir).expanduser(),
        display_dirs={
            "garage_ship": Path(args.garage_dir).expanduser(),
            "pumpkins": Path(args.pumpkins_dir).expanduser(),
        },
        mqtt_url=args.mqtt_url,
        session=args.session,
        mock_tracker=mock_tracker,
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
