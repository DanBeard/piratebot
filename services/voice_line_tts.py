"""
Voice Line TTS - Pre-generated voice line lookup.

Implements ITTSEngine interface but returns pre-generated audio from
the voice line database instead of synthesizing in real-time.
"""

import json
import logging
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np

from interfaces.tts_engine import ITTSEngine, TTSConfig, TTSResult, Viseme
from services.voice_line_db import VoiceLineDB, VoiceLine, SearchResult

logger = logging.getLogger(__name__)


class VoiceLineTTS(ITTSEngine):
    """
    TTS implementation using pre-generated voice lines.

    Instead of real-time synthesis, this looks up the best matching
    pre-generated audio file from the voice line database.

    Advantages:
    - Consistent voice quality (same pirate voice every time)
    - Fast playback (<100ms lookup vs 1-3s synthesis)
    - Guaranteed family-friendly content
    - Works offline after initial setup
    """

    def __init__(
        self,
        db_path: str = "data/voice_line_db",
        audio_base_path: str = "godot_project/assets/audio",
        sample_rate: int = 24000,
    ):
        """
        Initialize VoiceLineTTS.

        Args:
            db_path: Path to ChromaDB voice line database
            audio_base_path: Base path for audio files
            sample_rate: Expected sample rate of audio files
        """
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = sample_rate

        # Initialize voice line database
        self._db = VoiceLineDB(db_path=db_path)

        # Cache for loaded audio/visemes
        self._audio_cache: dict[str, np.ndarray] = {}
        self._viseme_cache: dict[str, list[Viseme]] = {}

        # Last selected line for reference
        self._last_selected: Optional[SearchResult] = None

        logger.info(f"VoiceLineTTS initialized: db={db_path}, audio={audio_base_path}")

    def _load_audio_file(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        Load audio file as numpy array.

        Args:
            audio_path: Path to WAV file

        Returns:
            Audio data as float32 array, or None if failed
        """
        try:
            with wave.open(str(audio_path), "rb") as wf:
                # Read audio data
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()

                # Convert to numpy array
                if sample_width == 2:  # 16-bit
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    audio = audio / 32768.0
                elif sample_width == 4:  # 32-bit
                    audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                    audio = audio / 2147483648.0
                else:
                    logger.error(f"Unsupported sample width: {sample_width}")
                    return None

                return audio

        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return None

    def _load_viseme_file(self, viseme_path: Path) -> list[Viseme]:
        """
        Load viseme data from JSON file.

        Args:
            viseme_path: Path to viseme JSON file

        Returns:
            List of Viseme objects
        """
        try:
            with open(viseme_path, "r") as f:
                data = json.load(f)

            visemes = []
            for cue in data.get("mouthCues", []):
                visemes.append(Viseme(
                    shape=cue["value"],
                    start_time=cue["start"],
                    end_time=cue["end"],
                ))

            return visemes

        except Exception as e:
            logger.warning(f"Failed to load viseme file {viseme_path}: {e}")
            return []

    def _get_audio_path(self, line: VoiceLine) -> Path:
        """Get the audio file path for a voice line."""
        if line.audio_file:
            return self.audio_base_path / line.audio_file

        # Default path based on category/id
        return self.audio_base_path / line.category / f"{line.id}.wav"

    def _get_viseme_path(self, line: VoiceLine) -> Path:
        """Get the viseme file path for a voice line."""
        if line.viseme_file:
            return self.audio_base_path / line.viseme_file

        # Default path based on category/id
        return self.audio_base_path / line.category / f"{line.id}.json"

    def search_lines(
        self,
        desired_text: str,
        category_hint: Optional[str] = None,
        costume_type: Optional[str] = None,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Search for voice lines matching the desired text.

        This is designed to be called by the LLM via tool-calling.

        Args:
            desired_text: What we want to say
            category_hint: Suggested category
            costume_type: Type of costume for tag matching
            n_results: Number of results to return

        Returns:
            List of dicts with id, text, category, similarity_score
        """
        results = self._db.search(
            query_text=desired_text,
            n_results=n_results,
            category=category_hint,
            tags=[costume_type] if costume_type else None,
        )

        return [
            {
                "id": r.line.id,
                "text": r.line.text,
                "category": f"{r.line.category}/{r.line.subcategory}" if r.line.subcategory else r.line.category,
                "similarity_score": round(r.similarity_score, 3),
            }
            for r in results
        ]

    def select_line(self, line_id: str) -> Optional[VoiceLine]:
        """
        Select a voice line by ID for playback.

        Args:
            line_id: The voice line ID to select

        Returns:
            VoiceLine if found, None otherwise
        """
        line = self._db.get_line(line_id)
        if line:
            self._db.mark_used(line_id)
            self._last_selected = SearchResult(line=line, similarity_score=1.0)
            logger.info(f"Selected voice line: {line_id}")
        return line

    def get_best_line(
        self,
        desired_text: str,
        category_hint: Optional[str] = None,
        costume_type: Optional[str] = None,
    ) -> Optional[VoiceLine]:
        """
        Get the best matching voice line with automatic fallback.

        This combines search and selection in one call.

        Args:
            desired_text: What we want to say
            category_hint: Suggested category
            costume_type: Type of costume

        Returns:
            Best matching VoiceLine
        """
        result = self._db.search_with_fallback(
            query_text=desired_text,
            category_hint=category_hint,
            costume_type=costume_type,
        )

        if result:
            self._db.mark_used(result.line.id)
            self._last_selected = result
            return result.line

        return None

    def synthesize(
        self,
        text: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSResult:
        """
        "Synthesize" speech by looking up pre-generated audio.

        The text parameter is used to find the best matching voice line.
        If the text exactly matches a line ID, that line is used directly.

        Args:
            text: Text to speak (used for search) or line ID prefixed with "id:"
            config: TTS configuration (mostly ignored for pre-generated audio)

        Returns:
            TTSResult with audio data
        """
        import time
        start_time = time.time()

        # Check if text is a line ID reference
        if text.startswith("id:"):
            line_id = text[3:]
            line = self._db.get_line(line_id)
            if line:
                self._db.mark_used(line_id)
                self._last_selected = SearchResult(line=line, similarity_score=1.0)
        else:
            # Search for best matching line
            line = self.get_best_line(text)

        if not line:
            logger.error(f"No voice line found for: {text[:50]}...")
            # Return silent audio
            return TTSResult(
                audio=np.zeros(self.sample_rate, dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=1.0,
                synthesis_time_ms=0,
            )

        # Load audio file
        audio_path = self._get_audio_path(line)

        # Check cache first
        if line.id in self._audio_cache:
            audio = self._audio_cache[line.id]
        else:
            audio = self._load_audio_file(audio_path)

            if audio is None:
                logger.warning(f"Audio file not found: {audio_path}")
                # Return silent audio
                return TTSResult(
                    audio=np.zeros(self.sample_rate, dtype=np.float32),
                    sample_rate=self.sample_rate,
                    duration_seconds=1.0,
                    synthesis_time_ms=0,
                )

            # Cache for future use
            self._audio_cache[line.id] = audio

        elapsed_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        logger.info(
            f"Voice line lookup: {line.id} ({duration:.2f}s audio in {elapsed_ms:.0f}ms)"
        )

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            synthesis_time_ms=elapsed_ms,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        config: Optional[TTSConfig] = None,
    ) -> Path:
        """
        Copy pre-generated audio to output path.

        Args:
            text: Text to speak or line ID
            output_path: Where to save the audio
            config: TTS configuration

        Returns:
            Path to the audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the audio data
        result = self.synthesize(text, config)

        # Save to file
        audio_int16 = (result.audio * 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path

    def generate_visemes(
        self,
        audio_path: Union[str, Path],
    ) -> list[Viseme]:
        """
        Get pre-generated visemes for the last selected line.

        For VoiceLineTTS, visemes are pre-computed and stored alongside
        the audio files. This method loads them from the JSON file.

        Args:
            audio_path: Path to audio file (used to find viseme file)

        Returns:
            List of Viseme objects
        """
        if not self._last_selected:
            logger.warning("No line selected, cannot get visemes")
            return []

        line = self._last_selected.line

        # Check cache
        if line.id in self._viseme_cache:
            return self._viseme_cache[line.id]

        # Load viseme file
        viseme_path = self._get_viseme_path(line)
        visemes = self._load_viseme_file(viseme_path)

        # Cache for future use
        self._viseme_cache[line.id] = visemes

        return visemes

    def get_visemes_for_line(self, line_id: str) -> list[Viseme]:
        """
        Get visemes for a specific line ID.

        Args:
            line_id: Voice line ID

        Returns:
            List of Viseme objects
        """
        # Check cache
        if line_id in self._viseme_cache:
            return self._viseme_cache[line_id]

        # Get the line
        line = self._db.get_line(line_id)
        if not line:
            return []

        # Load and cache
        viseme_path = self._get_viseme_path(line)
        visemes = self._load_viseme_file(viseme_path)
        self._viseme_cache[line_id] = visemes

        return visemes

    def get_available_voices(self) -> list[str]:
        """Return categories as 'voices'."""
        categories = self._db.get_categories()
        return list(categories.keys())

    def warmup(self) -> None:
        """Pre-load database and verify audio files."""
        self._db.warmup()

        # Check that audio files exist for at least some lines
        missing_count = 0
        checked_count = 0

        for cat_key, count in self._db.get_categories().items():
            parts = cat_key.split("/")
            category = parts[0]
            subcategory = parts[1] if len(parts) > 1 else None

            line = self._db.get_random_from_category(
                category, subcategory, exclude_recently_used=False
            )
            if line:
                checked_count += 1
                audio_path = self._get_audio_path(line)
                if not audio_path.exists():
                    missing_count += 1
                    logger.warning(f"Missing audio file: {audio_path}")

        if missing_count > 0:
            logger.warning(
                f"{missing_count}/{checked_count} categories have missing audio files. "
                f"Run tools/generate_voice_lines.py to generate them."
            )

        logger.info(f"VoiceLineTTS warmup complete: {self._db.count()} lines available")

    def cleanup(self) -> None:
        """Clear caches."""
        self._audio_cache.clear()
        self._viseme_cache.clear()
        logger.info("VoiceLineTTS caches cleared")

    def get_model_info(self) -> dict:
        """Return TTS engine information."""
        return {
            "name": "VoiceLineTTS",
            "type": "pre-generated",
            "database_path": str(self._db.db_path),
            "audio_path": str(self.audio_base_path),
            "total_lines": self._db.count(),
            "categories": self._db.get_categories(),
            "cached_audio": len(self._audio_cache),
            "cached_visemes": len(self._viseme_cache),
        }

    def get_last_selected(self) -> Optional[SearchResult]:
        """Return the last selected voice line result."""
        return self._last_selected

    def get_db(self) -> VoiceLineDB:
        """Get the underlying voice line database."""
        return self._db


# Testing function
def test_voice_line_tts():
    """Test VoiceLineTTS functionality."""
    # First, load voice lines from YAML
    from services.voice_line_db import VoiceLineDB

    db = VoiceLineDB(db_path="data/voice_line_db")

    yaml_path = Path("data/voice_lines.yaml")
    if yaml_path.exists():
        count = db.load_from_yaml(str(yaml_path))
        print(f"Loaded {count} voice lines from YAML")
    else:
        print("No voice_lines.yaml found, using existing database")

    # Create TTS instance
    tts = VoiceLineTTS()
    tts.warmup()

    # Test search
    print("\n--- Search Test ---")
    results = tts.search_lines(
        "That vampire costume is amazing!",
        category_hint="costume_reactions",
        costume_type="vampire",
    )
    for r in results:
        print(f"  [{r['similarity_score']:.3f}] {r['id']}: {r['text'][:50]}...")

    # Test best line selection
    print("\n--- Best Line Test ---")
    line = tts.get_best_line(
        "What a scary zombie!",
        costume_type="zombie",
    )
    if line:
        print(f"Selected: {line.id}")
        print(f"Text: {line.text}")

    # Test synthesize (will warn about missing audio files)
    print("\n--- Synthesize Test ---")
    result = tts.synthesize("A friendly greeting for a trick-or-treater")
    print(f"Audio duration: {result.duration_seconds:.2f}s")
    print(f"Lookup time: {result.synthesis_time_ms:.0f}ms")

    if tts.get_last_selected():
        selected = tts.get_last_selected()
        print(f"Used line: {selected.line.id}")
        print(f"Match score: {selected.similarity_score:.3f}")

    print("\nModel info:", tts.get_model_info())


if __name__ == "__main__":
    test_voice_line_tts()
