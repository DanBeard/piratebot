#!/usr/bin/env python3
"""
Voice Line Generator for Mac (Apple Silicon)

Enhanced voice line generator with MPS backend support and progress tracking.
Designed for safe stop/resume operation on Apple M-series chips.

Critical: Uses torch.float32 on MPS - float16 causes NaN errors in voice cloning.

Usage:
    python tools/generate_voice_lines_mac.py                    # Generate all lines
    python tools/generate_voice_lines_mac.py --category greetings  # Specific category
    python tools/generate_voice_lines_mac.py --reset-progress   # Clear progress and restart
    python tools/generate_voice_lines_mac.py --verify           # Verify existing files
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
    device: str = ""
    dtype: str = ""
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
                state.device = data.get("device", "")
                state.dtype = data.get("dtype", "")
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
            "device": self.device,
            "dtype": self.dtype,
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


def get_device() -> str:
    """Auto-detect best available device."""
    import torch

    if torch.backends.mps.is_available():
        logger.info("MPS backend available (Apple Silicon)")
        return "mps"
    elif torch.cuda.is_available():
        logger.info(f"CUDA backend available ({torch.cuda.get_device_name(0)})")
        return "cuda"

    logger.warning("No GPU backend available, using CPU (this will be slow)")
    return "cpu"


def get_dtype(device: str):
    """Get appropriate dtype for device.

    CRITICAL: MPS voice cloning requires float32 - float16 causes NaN errors.
    """
    import torch

    if device == "mps":
        logger.info("Using float32 (required for MPS voice cloning)")
        return torch.float32

    # CUDA can use float16 for faster inference
    return torch.float16


class VoiceLineGeneratorMac:
    """Generate audio files for voice lines using Qwen3-TTS with MPS support."""

    def __init__(
        self,
        voice_sample_path: str = "data/pirate_voice_sample.wav",
        output_dir: str = "godot_project/assets/audio",
        db_path: str = "data/voice_line_db",
        model_size: str = "1.7B",
        device: str = "auto",
        progress_path: str = "data/.voice_gen_progress.json",
    ):
        """
        Initialize voice line generator.

        Args:
            voice_sample_path: Path to pirate voice sample for cloning
            output_dir: Directory for generated audio files
            db_path: Path to ChromaDB database
            model_size: Qwen3-TTS model size ("1.7B" or "0.6B")
            device: Device to run on ("auto", "mps", "cuda", "cpu")
            progress_path: Path to progress tracking file
        """
        self.voice_sample_path = Path(voice_sample_path)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.model_size = model_size
        self.progress_path = Path(progress_path)

        # Auto-detect device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = device

        self.dtype = get_dtype(self.device)

        self._tts_model = None
        self._db: Optional[VoiceLineDB] = None
        self._shutdown_requested = False

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
        """Lazy-load Qwen3-TTS model with MPS-specific configuration."""
        if self._tts_model is not None:
            return True

        try:
            from qwen_tts import Qwen3TTSModel
            import torch

            logger.info(f"Loading Qwen3-TTS model ({self.model_size})...")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Dtype: {self.dtype}")

            # Map model size to model ID
            # - Base model: for voice cloning (generate_voice_clone)
            # - CustomVoice model: for instruction-based generation (generate_custom_voice)
            if self.voice_sample_path.exists():
                # Use Base model for voice cloning
                model_id = {
                    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                }.get(self.model_size, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            else:
                # Use CustomVoice model for instruction-based generation
                model_id = {
                    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # 0.6B doesn't have CustomVoice
                }.get(self.model_size, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

            # CRITICAL: Use float32 on MPS - float16/bfloat16 causes NaN errors
            if self.device == "mps":
                self._tts_model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map="mps",
                    dtype=torch.float32,  # CRITICAL: float16 fails on MPS
                )
            else:
                self._tts_model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                )

            # Store voice sample path for use during generation
            if self.voice_sample_path.exists():
                logger.info(f"Voice sample available: {self.voice_sample_path}")
                self._has_voice_sample = True
            else:
                logger.warning(
                    f"Voice sample not found: {self.voice_sample_path}. "
                    f"Using default voice."
                )
                self._has_voice_sample = False

            return True

        except ImportError:
            logger.error(
                "qwen-tts is not installed. Install with: pip install qwen-tts"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS: {e}")
            return False

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
        """Generate audio for a text using Qwen3-TTS."""
        if not self._ensure_tts_loaded():
            return self._generate_audio_fallback(text, output_path)

        try:
            import torch
            import soundfile as sf

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate audio using appropriate method
            if getattr(self, "_has_voice_sample", False):
                # Use voice cloning with reference audio (Base model)
                wavs, sr = self._tts_model.generate_voice_clone(
                    text=text,
                    language="English",
                    ref_audio=str(self.voice_sample_path),
                    x_vector_only_mode=True,  # Use speaker embedding only
                )
            else:
                # Use custom voice with pirate-style instruction (CustomVoice model)
                instruct = (
                    "Speak with a gruff pirate voice, enthusiastic and theatrical, "
                    "with hearty emphasis on pirate words like 'Arrr' and 'matey'."
                )
                wavs, sr = self._tts_model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker="Ryan",  # Male voice
                    instruct=instruct,
                )

            # Save to file
            sf.write(str(output_path), wavs[0], sr)

            # Clean up GPU memory after each generation (important for MPS)
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
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
        progress.device = self.device
        progress.dtype = str(self.dtype)
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
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to run TTS on",
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
        device=args.device,
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
