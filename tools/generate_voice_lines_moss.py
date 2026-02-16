#!/usr/bin/env python3
"""
Voice Line Generator using MOSS-TTS (CUDA / RTX 3080).

Generates pirate voice lines using MOSS-TTS models on NVIDIA GPUs.
Designed for offline batch generation - not real-time.

Supports three modes:
1. MOSS-TTS (8B) - Best quality, zero-shot voice cloning from reference audio
2. MOSS-TTS-Local (1.7B) - Lighter, fits easily in 10GB VRAM
3. MOSS-VoiceGenerator (1.7B) - Voice design from text description (no ref audio)

Requirements:
    - NVIDIA GPU with CUDA (tested on RTX 3080, 10GB VRAM)
    - MOSS-TTS installed: pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
    - transformers >= 5.0.0, torch >= 2.9.1

Usage:
    python tools/generate_voice_lines_moss.py                           # Generate all (1.7B local)
    python tools/generate_voice_lines_moss.py --model 8B                # Use 8B for best quality
    python tools/generate_voice_lines_moss.py --model voicegen          # Use VoiceGenerator
    python tools/generate_voice_lines_moss.py --category greetings      # Specific category
    python tools/generate_voice_lines_moss.py --test                    # Quick test (2 lines)
    python tools/generate_voice_lines_moss.py --verify                  # Verify existing files
    python tools/generate_voice_lines_moss.py --list                    # List lines and status
"""

import argparse
import importlib.util
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


# MOSS-TTS model IDs on HuggingFace
MOSS_MODELS = {
    "8B": "OpenMOSS-Team/MOSS-TTS",
    "1.7B": "OpenMOSS-Team/MOSS-TTS-Local",
    "voicegen": "OpenMOSS-Team/MOSS-VoiceGenerator",
}


@dataclass
class ProgressState:
    """Tracks generation progress for safe resume."""

    version: int = 1
    started_at: str = ""
    backend: str = "cuda"
    model: str = "1.7B"
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
                state.backend = data.get("backend", "cuda")
                state.model = data.get("model", "1.7B")
                state.completed_lines = data.get("completed_lines", [])
                state.failed_lines = data.get("failed_lines", [])
                state.current_line = data.get("current_line")
                state.stats = data.get("stats", {
                    "total": 0, "completed": 0, "avg_time_ms": 0, "times": [],
                })
                logger.info(
                    f"Loaded progress: {len(state.completed_lines)} completed, "
                    f"{len(state.failed_lines)} failed"
                )
                return state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load progress file: {e}, starting fresh")

        return cls(started_at=datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> None:
        """Save progress to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.stats.get("times"):
            self.stats["avg_time_ms"] = sum(self.stats["times"]) / len(self.stats["times"])

        data = {
            "version": self.version,
            "started_at": self.started_at,
            "backend": self.backend,
            "model": self.model,
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


def resolve_attn_implementation(device: str, dtype) -> str:
    """Pick the best attention implementation for the current hardware."""
    import torch

    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"CUDA available: {gpu_name} ({vram_gb:.1f} GB)")
            return True
        else:
            logger.error("CUDA not available")
            return False
    except ImportError:
        logger.error("PyTorch not installed")
        return False


class VoiceLineGeneratorMOSS:
    """Generate audio files for voice lines using MOSS-TTS on CUDA."""

    # Recommended generation hyperparameters per model variant
    GEN_PARAMS = {
        "8B": {
            "max_new_tokens": 4096,
            "audio_temperature": 1.7,
            "audio_top_p": 0.8,
            "audio_top_k": 25,
            "audio_repetition_penalty": 1.0,
        },
        "1.7B": {
            "max_new_tokens": 4096,
            "audio_temperature": 1.0,
            "audio_top_p": 0.95,
            "audio_top_k": 50,
            "audio_repetition_penalty": 1.1,
        },
        "voicegen": {
            "max_new_tokens": 4096,
            "audio_temperature": 1.5,
            "audio_top_p": 0.6,
            "audio_top_k": 50,
            "audio_repetition_penalty": 1.1,
        },
    }

    # Pirate voice description for VoiceGenerator
    PIRATE_VOICE_INSTRUCTION = (
        "Gruff, gravelly pirate voice, deep and commanding with a hearty, menacing tone "
        "in British English, radiating danger and swagger. "
        "West Country English accent from Bristol. "
        "Sounds like Geoffrey Rush as Barbossa - theatrical British pirate."
    )

    def __init__(
        self,
        voice_sample_path: str = "data/pirate_voice_sample.wav",
        output_dir: str = "godot_project/assets/audio",
        db_path: str = "data/voice_line_db",
        model_size: str = "1.7B",
        device: str = "cuda",
        progress_path: str = "data/.voice_gen_moss_progress.json",
    ):
        self.voice_sample_path = Path(voice_sample_path)
        self.output_dir = Path(output_dir)
        self.db_path = db_path
        self.model_size = model_size
        self.device = device
        self.progress_path = Path(progress_path)

        self._model = None
        self._processor = None
        self._sample_rate = None
        self._db: Optional[VoiceLineDB] = None
        self._shutdown_requested = False

        self.stats = {
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }

        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nShutdown requested, finishing current line...")
        self._shutdown_requested = True

    def _ensure_model_loaded(self) -> bool:
        """Lazy-load MOSS-TTS model."""
        if self._model is not None:
            return True

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            model_id = MOSS_MODELS.get(self.model_size)
            if not model_id:
                logger.error(f"Unknown model size: {self.model_size}")
                return False

            logger.info(f"Loading MOSS-TTS model: {model_id}")
            logger.info(f"  Device: {self.device}")

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            attn_impl = resolve_attn_implementation(self.device, dtype)
            logger.info(f"  Attention: {attn_impl}")
            logger.info(f"  Dtype: {dtype}")

            # Disable cuDNN SDPA backend (required by MOSS-TTS)
            if self.device == "cuda":
                torch.backends.cuda.enable_cudnn_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)

            # Load processor
            normalize = self.model_size == "voicegen"
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                **({"normalize_inputs": True} if normalize else {}),
            )
            self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(
                self.device
            )

            # Load model
            self._model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=dtype,
            ).to(self.device)
            self._model.eval()

            self._sample_rate = self._processor.model_config.sampling_rate

            # Log VRAM usage
            if self.device == "cuda":
                vram_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"  VRAM used: {vram_used:.1f} GB")

            logger.info(f"  Sample rate: {self._sample_rate} Hz")
            logger.info("  Model loaded successfully!")
            return True

        except ImportError as e:
            logger.error(
                f"Missing dependency: {e}\n"
                "Install MOSS-TTS:\n"
                "  git clone https://github.com/OpenMOSS/MOSS-TTS.git\n"
                "  cd MOSS-TTS\n"
                "  pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load MOSS-TTS: {e}")
            import traceback
            traceback.print_exc()
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

    def _generate_tts(self, text: str) -> Optional["torch.Tensor"]:
        """Generate audio tensor from text using MOSS-TTS or MOSS-TTS-Local.

        Uses voice cloning if a reference audio sample exists, otherwise plain TTS.
        """
        import torch

        # Build conversation with optional reference audio
        if self.voice_sample_path.exists():
            logger.info("  Using voice cloning from reference audio")
            user_msg = self._processor.build_user_message(
                text=text,
                reference=[str(self.voice_sample_path)],
            )
        else:
            user_msg = self._processor.build_user_message(text=text)

        conversations = [[user_msg]]

        with torch.no_grad():
            batch = self._processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            params = self.GEN_PARAMS.get(self.model_size, self.GEN_PARAMS["1.7B"])
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params,
            )

        for message in self._processor.decode(outputs):
            if message.audio_codes_list:
                return message.audio_codes_list[0]

        return None

    def _generate_voicegen(self, text: str) -> Optional["torch.Tensor"]:
        """Generate audio tensor from text using MOSS-VoiceGenerator.

        Creates voice from text description (no reference audio needed).
        """
        import torch

        user_msg = self._processor.build_user_message(
            text=text,
            instruction=self.PIRATE_VOICE_INSTRUCTION,
        )
        conversations = [[user_msg]]

        with torch.no_grad():
            batch = self._processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            params = self.GEN_PARAMS["voicegen"]
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params,
            )

        for message in self._processor.decode(outputs):
            if message.audio_codes_list:
                return message.audio_codes_list[0]

        return None

    def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generate audio for text and save to WAV file."""
        if not self._ensure_model_loaded():
            return False

        try:
            import torch
            import torchaudio

            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Generating: '{text[:60]}...'")

            # Generate audio tensor
            if self.model_size == "voicegen":
                audio = self._generate_voicegen(text)
            else:
                audio = self._generate_tts(text)

            if audio is None:
                logger.error("  No audio generated")
                return False

            # Save to WAV
            # audio is a 1D tensor; torchaudio expects [channels, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio = audio.cpu().float()

            torchaudio.save(str(output_path), audio, self._sample_rate)

            duration = audio.shape[-1] / self._sample_rate
            logger.info(f"  Saved: {output_path.name} ({duration:.1f}s)")

            # Free CUDA cache periodically
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return True

        except Exception as e:
            logger.error(f"  Audio generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_visemes(self, audio_path: Path, output_path: Path) -> bool:
        """Generate viseme timing data using Rhubarb Lip Sync."""
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
            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()

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

        if audio_path.exists() and viseme_path.exists() and not force:
            logger.debug(f"Skipping {line.id} (already exists)")
            self.stats["skipped"] += 1
            return True

        if not self.generate_audio(line.text, audio_path):
            self.stats["failed"] += 1
            return False

        if not self.generate_visemes(audio_path, viseme_path):
            self.stats["failed"] += 1
            return False

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
                            f"{category}_{subcategory}_{line_idx:03d}",
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
            eta_sec = (remaining * avg_time_ms) / 1000
            if eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
            else:
                eta_str = f"{int(eta_sec)}s"
        else:
            eta_str = "calculating..."

        print(
            f"\r  [{current}/{total}] ({percent:.0f}%) "
            f"ETA: {eta_str} | {line.id[:30]:<30}",
            end="", flush=True,
        )

    def generate_all(
        self,
        yaml_path: str = "data/voice_lines.yaml",
        category_filter: Optional[str] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> dict:
        """Generate audio for all voice lines with progress tracking."""
        lines = self.load_lines_from_yaml(yaml_path)

        if category_filter:
            lines = [l for l in lines if l.category == category_filter]

        if limit:
            lines = lines[:limit]
            logger.info(f"Test mode: limiting to {limit} lines")

        progress = ProgressState.load(self.progress_path)
        progress.backend = "cuda"
        progress.model = self.model_size
        progress.stats["total"] = len(lines)

        logger.info(f"Processing {len(lines)} voice lines with MOSS-TTS ({self.model_size})...")
        if progress.completed_lines:
            logger.info(
                f"Resuming: {len(progress.completed_lines)} already completed"
            )

        times = progress.stats.get("times", [])

        for i, line in enumerate(lines):
            if self._shutdown_requested:
                logger.info("Stopping gracefully...")
                progress.save(self.progress_path)
                break

            if line.id in progress.completed_lines and not force:
                self.stats["skipped"] += 1
                continue

            progress.current_line = line.id
            progress.save(self.progress_path)

            start = time.time()
            success = self.generate_line(line, force=force)
            elapsed = (time.time() - start) * 1000

            if success:
                progress.completed_lines.append(line.id)
                times.append(elapsed)
                progress.stats["times"] = times[-100:]
            else:
                progress.failed_lines.append({
                    "id": line.id,
                    "error": "generation failed",
                    "time": datetime.now(timezone.utc).isoformat(),
                })

            progress.current_line = None
            progress.save(self.progress_path)

            avg_time = sum(times) / len(times) if times else 0
            completed_count = len(
                [l for l in lines if l.id in progress.completed_lines]
            )
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
        """Print all voice lines with their generation status."""
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
        description="Generate voice line audio using MOSS-TTS (CUDA)"
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
        help="Path to voice sample for cloning (used with 8B and 1.7B models)",
    )
    parser.add_argument(
        "--model",
        default="1.7B",
        choices=["8B", "1.7B", "voicegen"],
        help=(
            "MOSS-TTS model variant: "
            "8B (best quality, needs 4-bit quant for 10GB VRAM), "
            "1.7B (lighter, fits easily), "
            "voicegen (voice from text description, no ref audio)"
        ),
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
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--progress-file",
        default="data/.voice_gen_moss_progress.json",
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

    # Check CUDA unless just listing/verifying
    if not args.list_lines and not args.verify:
        if args.device == "cuda" and not check_cuda_available():
            logger.error("CUDA required. Use --device cpu for CPU (slow).")
            sys.exit(1)

    generator = VoiceLineGeneratorMOSS(
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
