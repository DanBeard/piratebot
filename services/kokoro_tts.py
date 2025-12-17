"""
Kokoro TTS implementation.

Kokoro is an ultra-fast (82M params) text-to-speech model.
It can run at 90-210x real-time on consumer GPUs.
"""

import logging
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np

from interfaces.tts_engine import ITTSEngine, TTSConfig, TTSResult, Viseme

logger = logging.getLogger(__name__)


class KokoroTTS(ITTSEngine):
    """
    Text-to-Speech using Kokoro TTS.

    Kokoro is an 82M parameter model that generates high-quality speech
    at extremely fast speeds (90-210x real-time).

    This implementation supports multiple backends:
    1. kokoro-onnx (recommended for speed)
    2. Native Kokoro Python package
    3. Fallback to pyttsx3 if Kokoro is unavailable
    """

    # Available voice presets (subset - check kokoro docs for full list)
    VOICES = {
        # American English voices
        "af_bella": "American Female - Bella",
        "af_nicole": "American Female - Nicole",
        "af_sarah": "American Female - Sarah",
        "af_sky": "American Female - Sky",
        "am_adam": "American Male - Adam",
        "am_michael": "American Male - Michael",
        # British English voices
        "bf_emma": "British Female - Emma",
        "bf_isabella": "British Female - Isabella",
        "bm_george": "British Male - George",
        "bm_lewis": "British Male - Lewis",
    }

    def __init__(
        self,
        voice: str = "am_adam",
        output_dir: Union[str, Path] = "audio_output",
        device: str = "cuda",
        sample_rate: int = 24000,
    ):
        """
        Initialize Kokoro TTS.

        Args:
            voice: Voice preset name (e.g., "am_adam", "af_sky")
            output_dir: Directory for output audio files
            device: Device to run on ("cuda" or "cpu")
            sample_rate: Output sample rate
        """
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.device = device
        self.sample_rate = sample_rate

        self._model = None
        self._backend = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"KokoroTTS initialized: voice={voice}, device={device}")

    def _ensure_loaded(self) -> str:
        """
        Lazy-load TTS backend. Returns the backend name.
        """
        if self._model is not None:
            return self._backend

        # Try different backends in order of preference
        backends = [
            self._try_kokoro_onnx,
            self._try_kokoro_native,
            self._try_pyttsx3_fallback,
        ]

        for try_backend in backends:
            backend_name = try_backend()
            if backend_name:
                self._backend = backend_name
                logger.info(f"Using TTS backend: {backend_name}")
                return backend_name

        raise RuntimeError(
            "No TTS backend available. Install one of:\n"
            "  pip install kokoro-onnx\n"
            "  pip install kokoro\n"
            "  pip install pyttsx3 (fallback)"
        )

    def _try_kokoro_onnx(self) -> Optional[str]:
        """Try to load kokoro-onnx backend."""
        try:
            from kokoro_onnx import Kokoro

            logger.info("Loading Kokoro ONNX model...")
            self._model = Kokoro(self.voice)
            return "kokoro-onnx"
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to load kokoro-onnx: {e}")
            return None

    def _try_kokoro_native(self) -> Optional[str]:
        """Try to load native Kokoro backend."""
        try:
            import kokoro

            logger.info("Loading native Kokoro model...")
            # Native kokoro API may vary - adjust as needed
            self._model = kokoro.Pipeline(voice=self.voice)
            return "kokoro-native"
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to load native Kokoro: {e}")
            return None

    def _try_pyttsx3_fallback(self) -> Optional[str]:
        """Fallback to pyttsx3 (not as good but works offline)."""
        try:
            import pyttsx3

            logger.warning("Using pyttsx3 fallback (quality may be lower)")
            self._model = pyttsx3.init()

            # Try to set a suitable voice
            voices = self._model.getProperty("voices")
            for voice in voices:
                if "male" in voice.name.lower():
                    self._model.setProperty("voice", voice.id)
                    break

            # Adjust rate for pirate speech
            self._model.setProperty("rate", 150)

            return "pyttsx3"
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to load pyttsx3: {e}")
            return None

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
            TTSResult with audio data
        """
        backend = self._ensure_loaded()
        config = config or TTSConfig(voice=self.voice)
        start_time = time.time()

        if backend == "kokoro-onnx":
            audio = self._synthesize_kokoro_onnx(text, config)
        elif backend == "kokoro-native":
            audio = self._synthesize_kokoro_native(text, config)
        elif backend == "pyttsx3":
            audio = self._synthesize_pyttsx3(text, config)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        elapsed_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        logger.debug(
            f"Synthesized {duration:.2f}s audio in {elapsed_ms:.0f}ms "
            f"({duration / (elapsed_ms / 1000):.1f}x real-time)"
        )

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            synthesis_time_ms=elapsed_ms,
        )

    def _synthesize_kokoro_onnx(
        self, text: str, config: TTSConfig
    ) -> np.ndarray:
        """Synthesize using kokoro-onnx."""
        # kokoro-onnx returns audio as numpy array
        audio, _ = self._model.create(
            text,
            voice=config.voice or self.voice,
            speed=config.speed,
        )
        return audio

    def _synthesize_kokoro_native(
        self, text: str, config: TTSConfig
    ) -> np.ndarray:
        """Synthesize using native Kokoro."""
        # API may vary - adjust as needed
        result = self._model(text, speed=config.speed)
        return result.audio

    def _synthesize_pyttsx3(
        self, text: str, config: TTSConfig
    ) -> np.ndarray:
        """Synthesize using pyttsx3 (saves to file then loads)."""
        import tempfile

        # pyttsx3 can only save to file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self._model.save_to_file(text, temp_path)
            self._model.runAndWait()

            # Load the audio file
            with wave.open(temp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0  # Normalize to -1.0 to 1.0

            return audio
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        config: Optional[TTSConfig] = None,
    ) -> Path:
        """
        Synthesize speech and save to WAV file.

        Args:
            text: Text to convert to speech
            output_path: Output file path
            config: Optional TTS configuration

        Returns:
            Path to saved audio file
        """
        result = self.synthesize(text, config)
        output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as WAV file
        self._save_wav(result.audio, output_path, result.sample_rate)

        logger.info(f"Audio saved to {output_path}")
        return output_path

    def _save_wav(
        self,
        audio: np.ndarray,
        path: Path,
        sample_rate: int,
    ) -> None:
        """Save audio array to WAV file."""
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def get_available_voices(self) -> list[str]:
        """Return list of available voice presets."""
        return list(self.VOICES.keys())

    def generate_visemes(
        self,
        audio_path: Union[str, Path],
    ) -> list[Viseme]:
        """
        Generate viseme timing data from audio using Rhubarb.

        Args:
            audio_path: Path to audio file

        Returns:
            List of Viseme objects with timing
        """
        audio_path = Path(audio_path)

        # Check if Rhubarb is available
        try:
            result = subprocess.run(
                ["rhubarb", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("Rhubarb not available, skipping viseme generation")
                return []
        except FileNotFoundError:
            logger.warning("Rhubarb not installed, skipping viseme generation")
            return []

        # Run Rhubarb to generate viseme data
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
                logger.error(f"Rhubarb error: {result.stderr}")
                return []

            import json

            data = json.loads(result.stdout)
            visemes = []

            for cue in data.get("mouthCues", []):
                visemes.append(
                    Viseme(
                        shape=cue["value"],
                        start_time=cue["start"],
                        end_time=cue["end"],
                    )
                )

            logger.info(f"Generated {len(visemes)} visemes")
            return visemes

        except subprocess.TimeoutExpired:
            logger.error("Rhubarb timed out")
            return []
        except Exception as e:
            logger.error(f"Viseme generation error: {e}")
            return []

    def warmup(self) -> None:
        """Load model into memory."""
        self._ensure_loaded()
        logger.info(f"TTS warmup complete (backend: {self._backend})")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            if self._backend == "pyttsx3":
                try:
                    self._model.stop()
                except Exception:
                    pass

            self._model = None
            self._backend = None
            logger.info("TTS resources cleaned up")

    def get_model_info(self) -> dict:
        """Return TTS engine information."""
        return {
            "name": "Kokoro TTS",
            "backend": self._backend,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "available_voices": list(self.VOICES.keys()),
        }


# Convenience function for testing
def test_pirate_speech():
    """Test TTS with pirate phrases."""
    tts = KokoroTTS(voice="am_adam")  # Male voice for pirate
    tts.warmup()

    phrases = [
        "Arrr, welcome aboard me ship, ye scallywag!",
        "Shiver me timbers! That be a mighty fine costume!",
        "Blimey! A little vampire! Don't be biting me crew now!",
    ]

    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)

    for i, phrase in enumerate(phrases):
        print(f"\nSynthesizing: {phrase}")

        output_path = output_dir / f"pirate_{i}.wav"
        tts.synthesize_to_file(phrase, output_path)

        print(f"Saved to: {output_path}")

        # Generate visemes if Rhubarb is available
        visemes = tts.generate_visemes(output_path)
        if visemes:
            print(f"Generated {len(visemes)} visemes")
            for v in visemes[:5]:  # Show first 5
                print(f"  {v.start_time:.2f}-{v.end_time:.2f}: {v.shape}")

    print(f"\nModel info: {tts.get_model_info()}")
    tts.cleanup()


if __name__ == "__main__":
    test_pirate_speech()
