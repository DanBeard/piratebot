"""
Interface for controlling the 3D avatar.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .tts_engine import Viseme


class Expression(Enum):
    """Available avatar expressions."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SURPRISED = "surprised"
    ANGRY = "angry"
    SAD = "sad"
    LAUGH = "laugh"
    THINKING = "thinking"


class Animation(Enum):
    """Available avatar animations."""

    IDLE = "idle"
    TALKING = "talking"
    WAVE = "wave"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"
    POINT = "point"
    LAUGH = "laugh"


@dataclass
class GazeTarget:
    """Target for avatar gaze direction."""

    # Screen coordinates (0.0-1.0, with 0.5, 0.5 being center)
    x: float
    y: float
    # Optional: depth/distance
    z: float = 1.0


class IAvatarController(ABC):
    """
    Abstract interface for controlling the 3D avatar.

    Implementations:
    - GodotAvatarController: Control Godot-rendered avatar via WebSocket
    - (Future) PygameAvatarController: Simple 2D sprite-based fallback
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the avatar renderer.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the avatar renderer."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the avatar renderer."""
        pass

    @abstractmethod
    async def play_audio(
        self,
        audio_path: Union[str, Path],
        visemes: Optional[list[Viseme]] = None,
    ) -> None:
        """
        Play audio through the avatar with optional lip-sync.

        Args:
            audio_path: Path to WAV audio file
            visemes: Optional list of viseme timing data for lip-sync
        """
        pass

    @abstractmethod
    async def play_audio_with_lipsync(
        self,
        audio_path: Union[str, Path],
    ) -> None:
        """
        Play audio with automatic lip-sync generation.
        This method will generate visemes automatically if not provided.

        Args:
            audio_path: Path to WAV audio file
        """
        pass

    @abstractmethod
    async def set_expression(self, expression: Expression) -> None:
        """
        Set the avatar's facial expression.

        Args:
            expression: The expression to display
        """
        pass

    @abstractmethod
    async def play_animation(
        self,
        animation: Animation,
        loop: bool = False,
    ) -> None:
        """
        Play an animation on the avatar.

        Args:
            animation: The animation to play
            loop: Whether to loop the animation
        """
        pass

    @abstractmethod
    async def set_gaze(self, target: GazeTarget) -> None:
        """
        Set the avatar's gaze direction.

        Args:
            target: Where the avatar should look
        """
        pass

    async def look_at_screen_position(self, x: float, y: float) -> None:
        """
        Convenience method to look at a position on screen.

        Args:
            x: Horizontal position (0.0 = left, 1.0 = right)
            y: Vertical position (0.0 = top, 1.0 = bottom)
        """
        await self.set_gaze(GazeTarget(x=x, y=y))

    async def look_at_center(self) -> None:
        """Look at the center of the screen (default position)."""
        await self.set_gaze(GazeTarget(x=0.5, y=0.5))

    @abstractmethod
    async def stop_audio(self) -> None:
        """Stop any currently playing audio."""
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Reset avatar to default state (neutral expression, idle animation)."""
        pass

    def get_status(self) -> dict:
        """
        Optional: Get current avatar status.

        Returns:
            Dict with current expression, animation, playing audio, etc.
        """
        return {}
