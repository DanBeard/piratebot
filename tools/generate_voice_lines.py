#!/usr/bin/env python3
"""
Voice Line Generator - Generate audio files from voice_lines.yaml using Qwen3-TTS.

This script:
1. Reads voice line definitions from data/voice_lines.yaml
2. Uses Qwen3-TTS to generate WAV files with voice cloning
3. Runs Rhubarb Lip Sync to generate viseme timing data
4. Updates the ChromaDB voice line database with file paths

Usage:
    python tools/generate_voice_lines.py                    # Generate all lines
    python tools/generate_voice_lines.py --category greetings  # Specific category
    python tools/generate_voice_lines.py --list              # List lines
    python tools/generate_voice_lines.py --verify            # Verify existing files
"""

import argparse
import json
import logging
import subprocess
import sys
import wave
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


class VoiceLineGenerator:
    """Generate audio files for voice lines using Qwen3-TTS."""

    def __init__(
        self,
        voice_sample_path: str = "data/pirate_voice_sample.wav",
        output_dir: str = "godot_project/assets/audio",
        db_path: str = "data/voice_line_db",
        model_size: str = "1.7B",  # or "0.6B" for smaller VRAM
        device: str = "cuda",
    ):
        """
        Initialize voice line generator.

        Args:
            voice_sample_path: Path to pirate voice sample for cloning
            output_dir: Directory for generated audio files
            db_path: Path to ChromaDB database
            model_size: Qwen3-TTS model size ("1.7B" or "0.6B")
            device: Device to run on ("cuda" or "cpu")
        """
        self.voice_sample_path = Path(voice_sample_path)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.model_size = model_size
        self.device = device

        self._tts_model = None
        self._db: Optional[VoiceLineDB] = None

        # Statistics
        self.stats = {
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }

    def _ensure_tts_loaded(self) -> bool:
        """Lazy-load Qwen3-TTS model."""
        if self._tts_model is not None:
            return True

        try:
            from qwen_tts import QwenTTS

            logger.info(f"Loading Qwen3-TTS model ({self.model_size})...")

            # Map model size to model ID
            model_id = {
                "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            }.get(self.model_size, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

            self._tts_model = QwenTTS(model_id, device=self.device)

            # Load voice sample for cloning
            if self.voice_sample_path.exists():
                logger.info(f"Loading voice sample: {self.voice_sample_path}")
                self._tts_model.set_reference_audio(str(self.voice_sample_path))
            else:
                logger.warning(
                    f"Voice sample not found: {self.voice_sample_path}. "
                    f"Using default voice."
                )

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
        """
        Generate audio for a text using Qwen3-TTS.

        Args:
            text: Text to synthesize
            output_path: Path to save WAV file

        Returns:
            True if successful
        """
        if not self._ensure_tts_loaded():
            return self._generate_audio_fallback(text, output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate audio
            logger.info(f"Generating: {text[:50]}...")

            # Add pirate-style instruction for better prosody
            instruction = (
                "Speak with a gruff pirate voice, enthusiastic and theatrical, "
                "with hearty emphasis on pirate words like 'Arrr' and 'matey'."
            )

            audio = self._tts_model.synthesize(
                text,
                instruction=instruction,
            )

            # Save to file
            self._tts_model.save_audio(audio, str(output_path))

            logger.info(f"Saved: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return False

    def _generate_audio_fallback(self, text: str, output_path: Path) -> bool:
        """
        Fallback audio generation using pyttsx3 or espeak.

        This is used when Qwen3-TTS is not available.
        """
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
        """
        Generate viseme timing data using Rhubarb Lip Sync.

        Args:
            audio_path: Path to audio file
            output_path: Path to save JSON viseme file

        Returns:
            True if successful
        """
        # Check if Rhubarb is available
        try:
            result = subprocess.run(
                ["rhubarb", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("Rhubarb not available, skipping viseme generation")
                return self._generate_visemes_fallback(audio_path, output_path)
        except FileNotFoundError:
            logger.warning("Rhubarb not installed, using fallback visemes")
            return self._generate_visemes_fallback(audio_path, output_path)

        try:
            logger.info(f"Generating visemes for: {audio_path.name}")

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

            logger.info(f"Saved visemes: {output_path}")
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
        """
        Generate simple fallback visemes based on audio duration.

        Creates basic mouth movements without actual lip-sync analysis.
        """
        try:
            # Get audio duration
            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()

            # Generate simple viseme pattern
            # Alternate between mouth shapes
            visemes = []
            shapes = ["A", "B", "C", "D", "E", "F", "X"]  # Rhubarb shape codes
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

            logger.info(f"Generated fallback visemes: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Fallback viseme generation failed: {e}")
            return False

    def generate_line(self, line: VoiceLine, force: bool = False) -> bool:
        """
        Generate audio and visemes for a single voice line.

        Args:
            line: VoiceLine to generate
            force: Overwrite existing files

        Returns:
            True if successful
        """
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

    def generate_all(
        self,
        yaml_path: str = "data/voice_lines.yaml",
        category_filter: Optional[str] = None,
        force: bool = False,
    ) -> dict:
        """
        Generate audio for all voice lines.

        Args:
            yaml_path: Path to voice_lines.yaml
            category_filter: Only generate for this category
            force: Overwrite existing files

        Returns:
            Statistics dictionary
        """
        lines = self.load_lines_from_yaml(yaml_path)

        if category_filter:
            lines = [l for l in lines if l.category == category_filter]

        logger.info(f"Generating audio for {len(lines)} voice lines...")

        for i, line in enumerate(lines):
            logger.info(f"[{i+1}/{len(lines)}] {line.id}")
            self.generate_line(line, force=force)

        logger.info(
            f"Generation complete: "
            f"{self.stats['generated']} generated, "
            f"{self.stats['skipped']} skipped, "
            f"{self.stats['failed']} failed"
        )

        return self.stats

    def verify_files(self, yaml_path: str = "data/voice_lines.yaml") -> dict:
        """
        Verify that all voice lines have audio and viseme files.

        Returns:
            Dictionary with missing file information
        """
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
        description="Generate voice line audio using Qwen3-TTS"
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
        default="cuda",
        help="Device to run TTS on (cuda or cpu)",
    )

    args = parser.parse_args()

    generator = VoiceLineGenerator(
        voice_sample_path=args.voice_sample,
        output_dir=args.output,
        db_path=args.db,
        model_size=args.model,
        device=args.device,
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
        )


if __name__ == "__main__":
    main()
