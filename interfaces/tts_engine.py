"""
Interface for Text-to-Speech engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis."""

    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    # Output format
    sample_rate: int = 24000
    channels: int = 1


@dataclass
class TTSResult:
    """Result from TTS synthesis."""

    # Audio data as numpy array (float32, -1.0 to 1.0)
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    synthesis_time_ms: float


@dataclass
class Viseme:
    """A single viseme (mouth shape) with timing."""

    # Viseme type: "A", "E", "I", "O", "U", "rest", etc.
    shape: str
    # Start time in seconds from audio start
    start_time: float
    # End time in seconds
    end_time: float


class ITTSEngine(ABC):
    """
    Abstract interface for Text-to-Speech engines.

    Implementations:
    - KokoroTTS: Kokoro TTS (82M params, very fast)
    - (Future) CoquiTTS: Coqui XTTS-v2 (voice cloning)
    - (Future) BarkTTS: Bark (expressive, handles non-speech)
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            config: Optional TTS configuration

        Returns:
            TTSResult containing audio data and metadata
        """
        pass

    @abstractmethod
    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        config: Optional[TTSConfig] = None,
    ) -> Path:
        """
        Synthesize speech and save to a WAV file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the WAV file
            config: Optional TTS configuration

        Returns:
            Path to the saved audio file
        """
        pass

    def get_available_voices(self) -> list[str]:
        """
        Return list of available voice names.

        Returns:
            List of voice identifiers
        """
        return []

    def generate_visemes(
        self,
        audio_path: Union[str, Path],
    ) -> list[Viseme]:
        """
        Optional: Generate viseme timing data from audio file.
        This is typically done using Rhubarb Lip Sync.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of Viseme objects with timing information
        """
        return []

    def warmup(self) -> None:
        """
        Optional: Load model into memory for faster first inference.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional: Clean up resources.
        """
        pass

    def get_model_info(self) -> dict:
        """
        Optional: Return information about the TTS engine.
        """
        return {}
