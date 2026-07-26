"""
Portrait Avatar Controller — "haunted painting" WebSocket backend.

Serves the browser player at ws://host:port/ws and http://host:port/.
The browser renders layered PNG sprites: background, pirate body, mouth
viseme, eyes + pupils. Commands include play_audio, gaze, expression,
emote, and reset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

from interfaces.avatar_controller import (
    IAvatarController,
    Expression,
    Animation,
    GazeTarget,
    Viseme,
)

logger = logging.getLogger(__name__)


def _rhubarb_visemes(audio_path: Path) -> list[Viseme]:
    """Run Rhubarb and convert its mouth cues to Viseme objects."""
    try:
        result = subprocess.run(
            ["rhubarb", str(audio_path), "-f", "json", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Rhubarb not found. Install from "
            "https://github.com/DanielSWolf/rhubarb-lip-sync"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Rhubarb timed out") from exc

    if result.returncode != 0:
        raise RuntimeError(f"Rhubarb failed: {result.stderr}")

    data = json.loads(result.stdout)
    cues = data.get("mouthCues", [])
    return [
        Viseme(
            shape=cue["value"],
            start_time=float(cue["start"]),
            end_time=float(cue["end"]),
        )
        for cue in cues
    ]


class PortraitAvatarController(IAvatarController):
    """
    WebSocket controller for the portrait-mode browser player.

    The browser connects to this server. The Python orchestrator sends
    commands that the browser renders as layered 2D sprites.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9877,
        assets_dir: Union[str, Path] = "portrait_viewer",
        visemes_dir: Union[str, Path] = "data/parrotts_cache",
    ):
        self.host = host
        self.port = port
        self.assets_dir = Path(assets_dir)
        self.visemes_dir = Path(visemes_dir)
        self.visemes_dir.mkdir(parents=True, exist_ok=True)

        self._server: Optional[asyncio.Task] = None
        self._clients: set[Any] = set()
        self._connected = False
        self._http_server: Optional[asyncio.Task] = None

        logger.info(f"PortraitAvatarController initialized on ws://{host}:{port}")

    async def connect(self) -> bool:
        """Start the WebSocket + HTTP asset server."""
        import aiohttp
        from aiohttp import web

        async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self._clients.add(ws)
            self._connected = True
            logger.info("Portrait browser connected")
            try:
                async for _msg in ws:
                    # Browser can send heartbeats or state; we currently ignore.
                    pass
            finally:
                self._clients.discard(ws)
                if not self._clients:
                    self._connected = False
                logger.info("Portrait browser disconnected")
            return ws

        async def index_handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(self.assets_dir / "index.html")

        app = web.Application()
        app.router.add_get("/ws", websocket_handler)
        app.router.add_static(
            "/audio/",
            path=str(self.visemes_dir.parent / "parrotts_cache"),
            name="audio",
        )
        app.router.add_static("/", path=str(self.assets_dir), name="static")
        app.router.add_get("/", index_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        self._http_server = asyncio.create_task(site.start())
        self._runner = runner
        logger.info(f"Portrait HTTP server at http://{self.host}:{self.port}/")
        return True

    async def disconnect(self) -> None:
        """Stop servers and close all browser connections."""
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception as exc:
                logger.warning(f"Error closing browser ws: {exc}")
        self._clients.clear()

        if self._http_server:
            self._http_server.cancel()
            try:
                await self._http_server
            except asyncio.CancelledError:
                pass

        runner = getattr(self, "_runner", None)
        if runner:
            await runner.cleanup()

        self._connected = False
        logger.info("Portrait avatar disconnected")

    def is_connected(self) -> bool:
        return self._connected and bool(self._clients)

    async def _broadcast(self, command: dict) -> None:
        """Send a JSON command to all connected browser clients."""
        if not self._clients:
            logger.debug("No portrait clients connected; dropping command")
            return

        payload = json.dumps(command)
        dead: set[Any] = set()

        for ws in self._clients:
            try:
                await ws.send(payload)
            except Exception as exc:
                logger.warning(f"Failed to send to portrait client: {exc}")
                dead.add(ws)

        for ws in dead:
            self._clients.discard(ws)

    def _viseme_json(self, visemes: Optional[list[Viseme]]) -> Optional[list[dict]]:
        if not visemes:
            return None
        return [
            {"shape": v.shape, "start": v.start_time, "end": v.end_time}
            for v in visemes
        ]

    def _audio_url(self, audio_path: Union[str, Path]) -> str:
        """Convert a local path to a browser-served URL."""
        path = Path(audio_path)
        # The player is served from /, and /audio/ maps to the cache dir.
        return f"/audio/{path.name}"

    async def play_audio(
        self,
        audio_path: Union[str, Path],
        visemes: Optional[list[Viseme]] = None,
    ) -> None:
        await self._broadcast({
            "type": "play_audio",
            "audio_url": self._audio_url(audio_path),
            "visemes": self._viseme_json(visemes),
        })

    async def play_audio_with_lipsync(
        self,
        audio_path: Union[str, Path],
    ) -> None:
        path = Path(audio_path)
        viseme_path = self.visemes_dir / f"{path.stem}.visemes.json"

        visemes: Optional[list[Viseme]] = None
        if viseme_path.exists():
            try:
                data = json.loads(viseme_path.read_text())
                visemes = [
                    Viseme(shape=v["shape"], start_time=v["start"], end_time=v["end"])
                    for v in data
                ]
            except Exception as exc:
                logger.warning(f"Failed to load cached visemes {viseme_path}: {exc}")
        else:
            logger.warning(f"No cached visemes for {path.name}; lip sync unavailable")

        await self.play_audio(audio_path, visemes)

    async def set_expression(self, expression: Expression) -> None:
        await self._broadcast({
            "type": "set_expression",
            "expression": expression.value,
        })

    async def play_animation(
        self,
        animation: Animation,
        loop: bool = False,
    ) -> None:
        await self._broadcast({
            "type": "play_animation",
            "animation": animation.value,
            "loop": loop,
        })

    async def set_gaze(self, target: GazeTarget) -> None:
        await self._broadcast({
            "type": "set_gaze",
            "x": target.x,
            "y": target.y,
        })

    async def stop_audio(self) -> None:
        await self._broadcast({"type": "stop_audio"})

    async def reset(self) -> None:
        await self._broadcast({"type": "reset"})

    def get_status(self) -> dict:
        return {
            "connected": self._connected,
            "clients": len(self._clients),
            "host": self.host,
            "port": self.port,
        }
