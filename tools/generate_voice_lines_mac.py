#!/usr/bin/env python3
"""
Voice Line Generator for Mac (Apple Silicon)

Voice line generator using mlx-audio for native Apple Silicon performance.
Designed for safe stop/resume operation on Apple M-series chips.

Uses models from mlx-community for fast inference on MLX backend:
- Kokoro-82M: Fast TTS with 54+ voice presets
- CSM-1B: Voice cloning from reference audio

Usage:
    uv run python tools/generate_voice_lines_mac.py                    # Generate all lines
    uv run python tools/generate_voice_lines_mac.py --category greetings  # Specific category
    uv run python tools/generate_voice_lines_mac.py --reset-progress   # Clear progress and restart
    uv run python tools/generate_voice_lines_mac.py --verify           # Verify existing files
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.voice_line_db import VoiceLineDB, VoiceLine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """Tracks generation progress for safe resume."""

    version: int = 1
    started_at: str = ""
    backend: str = "mlx"  # Always MLX for Mac
    completed_lines: list[str] = field(default_factory=list)
    failed_lines: list[dict] = field(default_factory=list)
    current_line: Optional[str] = None
    stats: dict = field(default_factory=lambda: {
        "total": 0,
        "completed": 0,
        "avg_time_ms": 0,
        "times": [],
    })

    @classmethod
    def load(cls, path: Path) -> "ProgressState":
        """Load progress from file, or create new state."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                state = cls()
                state.version = data.get("version", 1)
                state.started_at = data.get("started_at", "")
                state.backend = data.get("backend", data.get("device", "mlx"))  # Migrate old format
                state.completed_lines = data.get("completed_lines", [])
                state.failed_lines = data.get("failed_lines", [])
                state.current_line = data.get("current_line")
                state.stats = data.get("stats", {"total": 0, "completed": 0, "avg_time_ms": 0, "times": []})
                logger.info(f"Loaded progress: {len(state.completed_lines)} completed, {len(state.failed_lines)} failed")
                return state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load progress file: {e}, starting fresh")

        return cls(started_at=datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> None:
        """Save progress to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update avg_time_ms from times list
        if self.stats.get("times"):
            self.stats["avg_time_ms"] = sum(self.stats["times"]) / len(self.stats["times"])

        data = {
            "version": self.version,
            "started_at": self.started_at,
            "backend": self.backend,
            "completed_lines": self.completed_lines,
            "failed_lines": self.failed_lines,
            "current_line": self.current_line,
            "stats": {
                "total": self.stats.get("total", 0),
                "completed": len(self.completed_lines),
                "avg_time_ms": self.stats.get("avg_time_ms", 0),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def check_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon required)."""
    try:
        import mlx.core as mx
        # Simple test to verify MLX works
        _ = mx.array([1, 2, 3])
        logger.info("MLX backend available (Apple Silicon)")
        return True
    except ImportError:
        logger.error("MLX not available - this script requires Apple Silicon Mac")
        return False
    except Exception as e:
        logger.error(f"MLX error: {e}")
        return False


class VoiceLineGeneratorMac:
    """Generate audio files for voice lines using mlx-audio on Apple Silicon."""

    def __init__(
        self,
        voice_sample_path: str = "data/pirate_voice_sample.wav",
        output_dir: str = "godot_project/assets/audio",
        db_path: str = "data/voice_line_db",
        model_size: str = "1.7B",
        progress_path: str = "data/.voice_gen_progress.json",
    ):
        """
        Initialize voice line generator.

        Args:
            voice_sample_path: Path to pirate voice sample for cloning
            output_dir: Directory for generated audio files
            db_path: Path to ChromaDB database
            model_size: Qwen3-TTS model size ("1.7B" or "0.6B")
            progress_path: Path to progress tracking file
        """
        self.voice_sample_path = Path(voice_sample_path)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.model_size = model_size
        self.progress_path = Path(progress_path)

        # Verify MLX is available
        if not check_mlx_available():
            raise RuntimeError("MLX not available - requires Apple Silicon Mac")

        self._tts_model = None
        self._db: Optional[VoiceLineDB] = None
        self._shutdown_requested = False
        self._has_voice_sample = False
        self._model_type = "base"  # "base" for voice cloning, "voice_design" for designed voice

        # Statistics
        self.stats = {
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nShutdown requested, finishing current line...")
        self._shutdown_requested = True

    def _ensure_tts_loaded(self) -> bool:
        """Lazy-load Qwen3-TTS model using mlx-audio for Apple Silicon.

        Uses different model variants based on availability of voice sample:
        - Base model with voice cloning if sample exists
        - VoiceDesign model with pirate voice description otherwise
        """
        if self._tts_model is not None:
            return True

        try:
            from mlx_audio.tts.utils import load_model

            # Check if voice sample exists for voice cloning
            if self.voice_sample_path.exists():
                # Use Base model for voice cloning
                model_id = {
                    "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                    "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                }.get(self.model_size, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
                self._has_voice_sample = True
                self._model_type = "base"
                logger.info(f"Voice sample available: {self.voice_sample_path}")
                logger.info(f"Loading Qwen3-TTS Base model for voice cloning...")
            else:
                # Use VoiceDesign model to create pirate voice from description
                model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
                self._has_voice_sample = False
                self._model_type = "voice_design"
                logger.info(
                    f"Voice sample not found: {self.voice_sample_path}. "
                    f"Using VoiceDesign to create pirate voice."
                )

            logger.info(f"  Model: {model_id}")
            self._tts_model = load_model(model_id)
            logger.info("  Model loaded successfully!")

            return True

        except ImportError:
            logger.error(
                "mlx-audio is not installed. Install with: uv sync --extra mac"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_ref_text_path(self) -> Path:
        """Get path to reference text file (transcript of voice sample)."""
        return self.voice_sample_path.with_suffix(".txt")

    def _load_reference_text(self) -> Optional[str]:
        """Load reference text (transcript) for voice cloning.

        The qwen3_tts model expects ref_text to be the transcript of the ref_audio.
        If not provided, the model will attempt to work without it.
        """
        ref_text_path = self._get_ref_text_path()
        if ref_text_path.exists():
            try:
                with open(ref_text_path, "r") as f:
                    ref_text = f.read().strip()
                logger.info(f"  Reference text loaded: '{ref_text[:50]}...'")
                return ref_text
            except Exception as e:
                logger.warning(f"Failed to load reference text: {e}")
        else:
            logger.info(
                f"  No reference text file found at {ref_text_path}. "
                f"Create one with the transcript of your voice sample for better cloning."
            )
        return None

    def _get_db(self) -> VoiceLineDB:
        """Get or create database instance."""
        if self._db is None:
            self._db = VoiceLineDB(db_path=self.db_path)
        return self._db

    def _get_audio_path(self, line: VoiceLine) -> Path:
        """Get output path for audio file."""
        return self.output_dir / line.category / f"{line.id}.wav"

    def _get_viseme_path(self, line: VoiceLine) -> Path:
        """Get output path for viseme file."""
        return self.output_dir / line.category / f"{line.id}.json"

    def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generate audio for a text using Qwen3-TTS via mlx-audio."""
        if not self._ensure_tts_loaded():
            return self._generate_audio_fallback(text, output_path)

        try:
            import mlx.core as mx
            import soundfile as sf
            import numpy as np

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate audio using appropriate method based on model type
            if self._model_type == "base" and self._has_voice_sample:
                # Use Base model with voice cloning
                # ref_audio is a file path (string), not an array
                ref_text = self._load_reference_text()
                results = list(self._tts_model.generate(
                    text=text,
                    ref_audio=str(self.voice_sample_path),
                    ref_text=ref_text,
                ))
            else:
                # Use VoiceDesign model with pirate voice description
                pirate_voice_instruct = (
                    "A gruff, weathered male pirate voice with a deep, gravelly tone. "
                    "Speaks with theatrical enthusiasm, rolling Rs, and dramatic pauses. "
                    "Sounds like an old sea captain with a hearty, booming delivery."
                )
                results = list(self._tts_model.generate_voice_design(
                    text=text,
                    language="English",
                    instruct=pirate_voice_instruct,
                ))

            # Combine all audio segments
            audio_segments = [r.audio for r in results if r.audio is not None]
            if not audio_segments:
                logger.error("No audio generated")
                return False

            # Concatenate segments and convert to numpy
            if len(audio_segments) == 1:
                audio = np.array(audio_segments[0])
            else:
                audio = np.concatenate([np.array(seg) for seg in audio_segments])

            # Get sample rate from model
            sr = self._tts_model.sample_rate

            # Save to file
            sf.write(str(output_path), audio, sr)

            # Clear MLX cache periodically to avoid memory buildup
            mx.clear_cache()

            return True

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_audio_fallback(self, text: str, output_path: Path) -> bool:
        """Fallback audio generation using pyttsx3."""
        logger.warning("Using fallback TTS (pyttsx3)")

        try:
            import pyttsx3

            output_path.parent.mkdir(parents=True, exist_ok=True)

            engine = pyttsx3.init()

            # Try to set a male voice
            voices = engine.getProperty("voices")
            for voice in voices:
                if "male" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break

            # Slower rate for dramatic pirate speech
            engine.setProperty("rate", 140)

            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

            return output_path.exists()

        except ImportError:
            logger.error("pyttsx3 not available. Install with: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            return False

    def generate_visemes(self, audio_path: Path, output_path: Path) -> bool:
        """Generate viseme timing data using Rhubarb Lip Sync."""
        # Check if Rhubarb is available
        try:
            result = subprocess.run(
                ["rhubarb", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return self._generate_visemes_fallback(audio_path, output_path)
        except FileNotFoundError:
            logger.debug("Rhubarb not installed, using fallback visemes")
            return self._generate_visemes_fallback(audio_path, output_path)

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
                timeout=60,
            )

            if result.returncode != 0:
                logger.error(f"Rhubarb error: {result.stderr}")
                return self._generate_visemes_fallback(audio_path, output_path)

            # Save viseme data
            viseme_data = json.loads(result.stdout)

            with open(output_path, "w") as f:
                json.dump(viseme_data, f, indent=2)

            return True

        except subprocess.TimeoutExpired:
            logger.error("Rhubarb timed out")
            return self._generate_visemes_fallback(audio_path, output_path)
        except Exception as e:
            logger.error(f"Viseme generation failed: {e}")
            return self._generate_visemes_fallback(audio_path, output_path)

    def _generate_visemes_fallback(
        self, audio_path: Path, output_path: Path
    ) -> bool:
        """Generate simple fallback visemes based on audio duration."""
        try:
            # Get audio duration
            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()

            # Generate simple viseme pattern
            visemes = []
            shapes = ["A", "B", "C", "D", "E", "F", "X"]
            time_per_shape = 0.15
            current_time = 0.0
            shape_idx = 0

            while current_time < duration:
                end_time = min(current_time + time_per_shape, duration)
                visemes.append({
                    "start": round(current_time, 3),
                    "end": round(end_time, 3),
                    "value": shapes[shape_idx % len(shapes)],
                })
                current_time = end_time
                shape_idx += 1

            viseme_data = {
                "metadata": {
                    "duration": duration,
                    "generated": "fallback",
                },
                "mouthCues": visemes,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(viseme_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Fallback viseme generation failed: {e}")
            return False

    def generate_line(self, line: VoiceLine, force: bool = False) -> bool:
        """Generate audio and visemes for a single voice line."""
        audio_path = self._get_audio_path(line)
        viseme_path = self._get_viseme_path(line)

        # Check if already exists
        if audio_path.exists() and viseme_path.exists() and not force:
            logger.debug(f"Skipping {line.id} (already exists)")
            self.stats["skipped"] += 1
            return True

        # Generate audio
        if not self.generate_audio(line.text, audio_path):
            self.stats["failed"] += 1
            return False

        # Generate visemes
        if not self.generate_visemes(audio_path, viseme_path):
            self.stats["failed"] += 1
            return False

        # Update database with file paths
        line.audio_file = str(audio_path.relative_to(self.output_dir))
        line.viseme_file = str(viseme_path.relative_to(self.output_dir))

        db = self._get_db()
        db.add_line(line)

        self.stats["generated"] += 1
        return True

    def load_lines_from_yaml(self, yaml_path: str) -> list[VoiceLine]:
        """Load voice line definitions from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        lines = []
        line_idx = 0

        for category, subcategories in data.get("voice_lines", {}).items():
            for subcategory, entries in subcategories.items():
                for entry in entries:
                    if isinstance(entry, str):
                        line_id = f"{category}_{subcategory}_{line_idx:03d}"
                        lines.append(VoiceLine(
                            id=line_id,
                            text=entry,
                            category=category,
                            subcategory=subcategory,
                        ))
                    elif isinstance(entry, dict):
                        line_id = entry.get(
                            "id",
                            f"{category}_{subcategory}_{line_idx:03d}"
                        )
                        lines.append(VoiceLine(
                            id=line_id,
                            text=entry["text"],
                            category=category,
                            subcategory=subcategory,
                            tags=entry.get("tags", []),
                            emotion=entry.get("emotion", "neutral"),
                        ))
                    line_idx += 1

        return lines

    def _print_progress(
        self,
        current: int,
        total: int,
        line: VoiceLine,
        elapsed_ms: float,
        avg_time_ms: float,
    ) -> None:
        """Print progress with ETA."""
        percent = (current / total) * 100
        remaining = total - current

        if avg_time_ms > 0:
            eta_ms = remaining * avg_time_ms
            eta_sec = eta_ms / 1000
            if eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
            else:
                eta_str = f"{int(eta_sec)}s"
        else:
            eta_str = "calculating..."

        # Clear line and print progress
        print(f"\r  [{current}/{total}] ({percent:.0f}%) ETA: {eta_str} | {line.id[:30]:<30}", end="", flush=True)

    def generate_all(
        self,
        yaml_path: str = "data/voice_lines.yaml",
        category_filter: Optional[str] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> dict:
        """Generate audio for all voice lines with progress tracking.

        Args:
            yaml_path: Path to voice_lines.yaml
            category_filter: Only process this category
            force: Overwrite existing files
            limit: Maximum number of lines to generate (for testing)
        """
        lines = self.load_lines_from_yaml(yaml_path)

        if category_filter:
            lines = [l for l in lines if l.category == category_filter]

        if limit:
            lines = lines[:limit]
            logger.info(f"Test mode: limiting to {limit} lines")

        # Load or create progress state
        progress = ProgressState.load(self.progress_path)
        progress.backend = "mlx"
        progress.stats["total"] = len(lines)

        logger.info(f"Processing {len(lines)} voice lines...")
        if progress.completed_lines:
            logger.info(f"Resuming from previous progress: {len(progress.completed_lines)} already completed")

        times = progress.stats.get("times", [])

        for i, line in enumerate(lines):
            # Check for shutdown request
            if self._shutdown_requested:
                logger.info("Stopping gracefully...")
                progress.save(self.progress_path)
                break

            # Skip completed lines (unless force)
            if line.id in progress.completed_lines and not force:
                self.stats["skipped"] += 1
                continue

            # Mark current line (survives CTRL+C)
            progress.current_line = line.id
            progress.save(self.progress_path)

            # Generate
            start = time.time()
            success = self.generate_line(line, force=force)
            elapsed = (time.time() - start) * 1000

            # Update progress
            if success:
                progress.completed_lines.append(line.id)
                times.append(elapsed)
                progress.stats["times"] = times[-100:]  # Keep last 100 for avg
            else:
                progress.failed_lines.append({
                    "id": line.id,
                    "error": "generation failed",
                    "time": datetime.now(timezone.utc).isoformat(),
                })

            progress.current_line = None
            progress.save(self.progress_path)

            # Calculate avg time
            avg_time = sum(times) / len(times) if times else 0

            # Show progress
            completed_count = len([l for l in lines if l.id in progress.completed_lines])
            self._print_progress(completed_count, len(lines), line, elapsed, avg_time)

        print()  # Newline after progress

        logger.info(
            f"Generation complete: "
            f"{self.stats['generated']} generated, "
            f"{self.stats['skipped']} skipped, "
            f"{self.stats['failed']} failed"
        )

        return self.stats

    def verify_files(self, yaml_path: str = "data/voice_lines.yaml") -> dict:
        """Verify that all voice lines have audio and viseme files."""
        lines = self.load_lines_from_yaml(yaml_path)

        missing_audio = []
        missing_visemes = []
        valid = []

        for line in lines:
            audio_path = self._get_audio_path(line)
            viseme_path = self._get_viseme_path(line)

            if not audio_path.exists():
                missing_audio.append(line.id)
            elif not viseme_path.exists():
                missing_visemes.append(line.id)
            else:
                valid.append(line.id)

        return {
            "total": len(lines),
            "valid": len(valid),
            "missing_audio": missing_audio,
            "missing_visemes": missing_visemes,
        }

    def list_lines(
        self,
        yaml_path: str = "data/voice_lines.yaml",
        category_filter: Optional[str] = None,
    ) -> None:
        """Print all voice lines."""
        lines = self.load_lines_from_yaml(yaml_path)

        if category_filter:
            lines = [l for l in lines if l.category == category_filter]

        current_category = None
        for line in lines:
            cat_key = f"{line.category}/{line.subcategory}"
            if cat_key != current_category:
                current_category = cat_key
                print(f"\n[{current_category}]")

            audio_path = self._get_audio_path(line)
            status = "OK" if audio_path.exists() else "MISSING"
            print(f"  [{status}] {line.id}: {line.text[:60]}...")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate voice line audio using Qwen3-TTS (Mac/Apple Silicon)"
    )
    parser.add_argument(
        "--yaml",
        default="data/voice_lines.yaml",
        help="Path to voice_lines.yaml",
    )
    parser.add_argument(
        "--output",
        default="godot_project/assets/audio",
        help="Output directory for audio files",
    )
    parser.add_argument(
        "--db",
        default="data/voice_line_db",
        help="Path to ChromaDB database",
    )
    parser.add_argument(
        "--voice-sample",
        default="data/pirate_voice_sample.wav",
        help="Path to voice sample for cloning",
    )
    parser.add_argument(
        "--model",
        default="1.7B",
        choices=["1.7B", "0.6B"],
        help="Qwen3-TTS model size",
    )
    parser.add_argument(
        "--category",
        help="Only generate for this category",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_lines",
        help="List voice lines and their status",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing audio files",
    )
    parser.add_argument(
        "--progress-file",
        default="data/.voice_gen_progress.json",
        help="Path to progress tracking file",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Clear progress file and start fresh",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: generate only first 2 lines to verify setup",
    )

    args = parser.parse_args()

    # Handle progress reset
    progress_path = Path(args.progress_file)
    if args.reset_progress and progress_path.exists():
        progress_path.unlink()
        logger.info("Progress file cleared")

    generator = VoiceLineGeneratorMac(
        voice_sample_path=args.voice_sample,
        output_dir=args.output,
        db_path=args.db,
        model_size=args.model,
        progress_path=args.progress_file,
    )

    if args.list_lines:
        generator.list_lines(args.yaml, args.category)
    elif args.verify:
        result = generator.verify_files(args.yaml)
        print(f"\nVerification Results:")
        print(f"  Total lines: {result['total']}")
        print(f"  Valid: {result['valid']}")
        print(f"  Missing audio: {len(result['missing_audio'])}")
        print(f"  Missing visemes: {len(result['missing_visemes'])}")

        if result["missing_audio"]:
            print(f"\nMissing audio files:")
            for line_id in result["missing_audio"][:10]:
                print(f"  - {line_id}")
            if len(result["missing_audio"]) > 10:
                print(f"  ... and {len(result['missing_audio']) - 10} more")
    else:
        generator.generate_all(
            yaml_path=args.yaml,
            category_filter=args.category,
            force=args.force,
            limit=2 if args.test else None,
        )


if __name__ == "__main__":
    main()
