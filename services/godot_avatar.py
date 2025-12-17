"""
Godot Avatar Controller implementation.

Controls a 3D avatar rendered in Godot via WebSocket communication.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Union

from interfaces.avatar_controller import (
    IAvatarController,
    Expression,
    Animation,
    GazeTarget,
)
from interfaces.tts_engine import Viseme

logger = logging.getLogger(__name__)


class GodotAvatarController(IAvatarController):
    """
    Controls a Godot-rendered 3D avatar via WebSocket.

    The Godot project runs a WebSocket server that receives JSON commands
    to control the avatar's expressions, animations, lip-sync, and gaze.

    Protocol:
    - Commands are JSON objects with a "type" field
    - Godot sends acknowledgment responses
    - Audio files are referenced by path (Godot loads them directly)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9876,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 5,
    ):
        """
        Initialize Godot avatar controller.

        Args:
            host: Godot WebSocket server host
            port: Godot WebSocket server port
            reconnect_delay: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
        """
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._websocket = None
        self._connected = False
        self._message_id = 0

        logger.info(f"GodotAvatarController initialized: ws://{host}:{port}")

    def _get_uri(self) -> str:
        """Get WebSocket URI."""
        return f"ws://{self.host}:{self.port}"

    def _next_message_id(self) -> int:
        """Get next message ID for tracking responses."""
        self._message_id += 1
        return self._message_id

    async def _send_command(
        self,
        command_type: str,
        data: Optional[dict] = None,
        wait_response: bool = True,
    ) -> Optional[dict]:
        """
        Send a command to Godot and optionally wait for response.

        Args:
            command_type: Type of command (e.g., "play_audio", "set_expression")
            data: Command data
            wait_response: Whether to wait for acknowledgment

        Returns:
            Response data if wait_response is True
        """
        if not self._connected:
            raise RuntimeError("Not connected to Godot avatar")

        message = {
            "id": self._next_message_id(),
            "type": command_type,
            **(data or {}),
        }

        try:
            await self._websocket.send(json.dumps(message))
            logger.debug(f"Sent command: {command_type}")

            if wait_response:
                response_text = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=10.0,
                )
                response = json.loads(response_text)
                logger.debug(f"Received response: {response}")
                return response

            return None

        except asyncio.TimeoutError:
            logger.error(f"Command timed out: {command_type}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {e}")
            raise

    async def connect(self) -> bool:
        """
        Connect to the Godot WebSocket server.

        Returns:
            True if connection successful
        """
        try:
            import websockets

            attempts = 0
            while attempts < self.max_reconnect_attempts:
                try:
                    logger.info(f"Connecting to Godot at {self._get_uri()}...")
                    self._websocket = await websockets.connect(
                        self._get_uri(),
                        ping_interval=20,
                        ping_timeout=20,
                    )
                    self._connected = True
                    logger.info("Connected to Godot avatar")

                    # Send handshake
                    response = await self._send_command("handshake", {
                        "client": "PirateBot",
                        "version": "1.0",
                    })

                    if response and response.get("status") == "ok":
                        logger.info("Handshake successful")
                        return True

                except (ConnectionRefusedError, OSError) as e:
                    attempts += 1
                    logger.warning(
                        f"Connection attempt {attempts} failed: {e}"
                    )
                    if attempts < self.max_reconnect_attempts:
                        await asyncio.sleep(self.reconnect_delay)

            logger.error("Failed to connect to Godot after max attempts")
            return False

        except ImportError:
            raise ImportError(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )

    async def disconnect(self) -> None:
        """Close connection to Godot."""
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

            self._websocket = None
            self._connected = False
            logger.info("Disconnected from Godot avatar")

    def is_connected(self) -> bool:
        """Check if connected to Godot."""
        return self._connected and self._websocket is not None

    async def play_audio(
        self,
        audio_path: Union[str, Path],
        visemes: Optional[list[Viseme]] = None,
    ) -> None:
        """
        Play audio with optional lip-sync.

        Args:
            audio_path: Path to WAV audio file
            visemes: Optional viseme timing data
        """
        audio_path = Path(audio_path).absolute()

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Convert visemes to serializable format
        viseme_data = None
        if visemes:
            viseme_data = [
                {
                    "shape": v.shape,
                    "start": v.start_time,
                    "end": v.end_time,
                }
                for v in visemes
            ]

        await self._send_command(
            "play_audio",
            {
                "path": str(audio_path),
                "visemes": viseme_data,
            },
        )

    async def play_audio_with_lipsync(
        self,
        audio_path: Union[str, Path],
    ) -> None:
        """
        Play audio with automatic lip-sync generation.

        The visemes are generated using Rhubarb if available,
        or Godot will use amplitude-based lip-sync as fallback.
        """
        audio_path = Path(audio_path)

        # Try to generate visemes with Rhubarb
        visemes = self._generate_visemes(audio_path)

        await self.play_audio(audio_path, visemes)

    def _generate_visemes(self, audio_path: Path) -> list[Viseme]:
        """
        Generate visemes from audio file using Rhubarb.
        """
        import subprocess

        try:
            result = subprocess.run(
                [
                    "rhubarb",
                    str(audio_path),
                    "-f", "json",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f"Rhubarb failed: {result.stderr}")
                return []

            import json as json_module

            data = json_module.loads(result.stdout)
            visemes = []

            for cue in data.get("mouthCues", []):
                visemes.append(
                    Viseme(
                        shape=cue["value"],
                        start_time=cue["start"],
                        end_time=cue["end"],
                    )
                )

            return visemes

        except FileNotFoundError:
            logger.warning("Rhubarb not installed")
            return []
        except Exception as e:
            logger.warning(f"Viseme generation failed: {e}")
            return []

    async def set_expression(self, expression: Expression) -> None:
        """Set the avatar's facial expression."""
        await self._send_command(
            "set_expression",
            {"expression": expression.value},
        )

    async def play_animation(
        self,
        animation: Animation,
        loop: bool = False,
    ) -> None:
        """Play an animation on the avatar."""
        await self._send_command(
            "play_animation",
            {
                "animation": animation.value,
                "loop": loop,
            },
        )

    async def set_gaze(self, target: GazeTarget) -> None:
        """Set the avatar's gaze direction."""
        await self._send_command(
            "set_gaze",
            {
                "x": target.x,
                "y": target.y,
                "z": target.z,
            },
        )

    async def stop_audio(self) -> None:
        """Stop any currently playing audio."""
        await self._send_command("stop_audio")

    async def reset(self) -> None:
        """Reset avatar to default state."""
        await self._send_command("reset")

    def get_status(self) -> dict:
        """Get current connection status."""
        return {
            "connected": self._connected,
            "host": self.host,
            "port": self.port,
            "message_count": self._message_id,
        }


# Mock controller for testing without Godot
class MockAvatarController(IAvatarController):
    """
    Mock avatar controller for testing without Godot.
    Logs all commands instead of sending them.
    """

    def __init__(self):
        self._connected = False
        logger.info("MockAvatarController initialized")

    async def connect(self) -> bool:
        self._connected = True
        logger.info("[MOCK] Connected to avatar")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("[MOCK] Disconnected from avatar")

    def is_connected(self) -> bool:
        return self._connected

    async def play_audio(
        self,
        audio_path: Union[str, Path],
        visemes: Optional[list[Viseme]] = None,
    ) -> None:
        viseme_count = len(visemes) if visemes else 0
        logger.info(f"[MOCK] Playing audio: {audio_path} ({viseme_count} visemes)")

    async def play_audio_with_lipsync(
        self,
        audio_path: Union[str, Path],
    ) -> None:
        logger.info(f"[MOCK] Playing audio with lipsync: {audio_path}")

    async def set_expression(self, expression: Expression) -> None:
        logger.info(f"[MOCK] Set expression: {expression.value}")

    async def play_animation(
        self,
        animation: Animation,
        loop: bool = False,
    ) -> None:
        logger.info(f"[MOCK] Play animation: {animation.value} (loop={loop})")

    async def set_gaze(self, target: GazeTarget) -> None:
        logger.info(f"[MOCK] Set gaze: ({target.x}, {target.y}, {target.z})")

    async def stop_audio(self) -> None:
        logger.info("[MOCK] Stop audio")

    async def reset(self) -> None:
        logger.info("[MOCK] Reset avatar")


# Test function
async def test_avatar_connection():
    """Test avatar controller connection."""
    controller = GodotAvatarController()

    try:
        connected = await controller.connect()

        if connected:
            print("Connected to Godot!")

            # Test some commands
            await controller.set_expression(Expression.HAPPY)
            await controller.play_animation(Animation.WAVE)
            await controller.look_at_center()

            print("Commands sent successfully")

        else:
            print("Failed to connect. Is Godot running?")

    finally:
        await controller.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_avatar_connection())
