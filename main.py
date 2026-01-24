#!/usr/bin/env python3
"""
PirateBot - Interactive Halloween Pirate Decoration

Main orchestrator that coordinates all components:
- Webcam person detection
- Vision model for costume description
- LLM for pirate banter generation (with tool-calling for voice line selection)
- Pre-generated voice line playback (or real-time TTS fallback)
- 3D avatar control with lip-sync
"""

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Interfaces
from interfaces import (
    IDetector,
    IVisionModel,
    ILanguageModel,
    ITTSEngine,
    IAvatarController,
    Detection,
)

# Voice line imports
from services.voice_line_tts import VoiceLineTTS
from services.voice_line_db import VoiceLineDB
from services.voice_line_search import (
    VoiceLineToolHandler,
    VoiceLineSelection,
    get_voice_line_system_prompt,
    create_costume_user_prompt,
)
from services.ollama_llm import ToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("piratebot")


class PirateBot:
    """Main orchestrator for the interactive pirate decoration."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize PirateBot with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False

        # Components (will be initialized in setup())
        self.detector: Optional[IDetector] = None
        self.vlm: Optional[IVisionModel] = None
        self.llm: Optional[ILanguageModel] = None
        self.tts: Optional[ITTSEngine] = None
        self.avatar: Optional[IAvatarController] = None

        # Voice line system
        self.voice_line_tts: Optional[VoiceLineTTS] = None
        self.voice_line_handler: Optional[VoiceLineToolHandler] = None
        self.use_voice_lines = self.config.get("voice_lines", {}).get("enabled", True)

        # Webcam
        self.cap: Optional[cv2.VideoCapture] = None

        # State tracking
        self.last_interaction_time: dict[int, float] = {}  # track_id -> timestamp
        self.cooldown_seconds = self.config["detector"]["cooldown_seconds"]

        # Load prompts (different for voice line mode vs real-time TTS)
        self.system_prompt = self.config["prompts"]["system"]
        self.user_template = self.config["prompts"]["user_template"]

        logger.info("PirateBot initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def setup(self) -> None:
        """Initialize all components."""
        logger.info("Setting up PirateBot components...")

        # Import and instantiate implementations
        from services.yolo_detector import YoloDetector
        from services.moondream_vlm import MoondreamVLM
        from services.ollama_llm import OllamaLLM
        from services.kokoro_tts import KokoroTTS
        from services.godot_avatar import GodotAvatarController

        # Initialize detector
        self.detector = YoloDetector(
            model=self.config["detector"]["model"],
            confidence_threshold=self.config["detector"]["confidence_threshold"],
        )
        self.detector.warmup()

        # Initialize VLM
        self.vlm = MoondreamVLM(
            model_id=self.config["vlm"]["model_id"],
            device=f"cuda:{self.config['gpu']['vlm']}",
        )
        self.vlm.warmup()

        # Initialize LLM
        self.llm = OllamaLLM(
            base_url=self.config["llm"]["base_url"],
            model=self.config["llm"]["model"],
        )
        self.llm.warmup()

        # Initialize Voice Line System (if enabled)
        if self.use_voice_lines:
            await self._setup_voice_lines()
        else:
            # Fall back to real-time TTS
            self.tts = KokoroTTS(
                voice=self.config["tts"]["voice"],
                output_dir=self.config["tts"]["output_dir"],
            )
            self.tts.warmup()

        # Initialize Avatar
        self.avatar = GodotAvatarController(
            host=self.config["avatar"]["host"],
            port=self.config["avatar"]["port"],
        )
        await self.avatar.connect()

        # Initialize webcam
        self._setup_webcam()

        logger.info("All components initialized")

    async def _setup_voice_lines(self) -> None:
        """Initialize the voice line system."""
        vl_config = self.config.get("voice_lines", {})

        logger.info("Setting up voice line system...")

        # Initialize VoiceLineDB and load lines from YAML
        db_path = vl_config.get("db_path", "data/voice_line_db")
        yaml_path = vl_config.get("yaml_path", "data/voice_lines.yaml")
        audio_path = vl_config.get("audio_path", "godot_project/assets/audio")

        db = VoiceLineDB(
            db_path=db_path,
            recently_used_cooldown=vl_config.get("recently_used_cooldown", 300),
        )

        # Load voice lines from YAML if database is empty
        if db.count() == 0 and Path(yaml_path).exists():
            count = db.load_from_yaml(yaml_path)
            logger.info(f"Loaded {count} voice lines from {yaml_path}")

        # Initialize VoiceLineTTS
        self.voice_line_tts = VoiceLineTTS(
            db_path=db_path,
            audio_base_path=audio_path,
        )
        self.voice_line_tts.warmup()

        # Initialize tool handler for LLM
        self.voice_line_handler = VoiceLineToolHandler(self.voice_line_tts)

        # Update system prompt for tool-calling mode
        if vl_config.get("use_tool_calling", True):
            self.system_prompt = get_voice_line_system_prompt()

        logger.info(f"Voice line system ready: {db.count()} lines available")

    def _setup_webcam(self) -> None:
        """Initialize webcam capture."""
        webcam_config = self.config["webcam"]
        self.cap = cv2.VideoCapture(webcam_config["device_id"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, webcam_config["fps"])

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam {webcam_config['device_id']}")

        logger.info(
            f"Webcam initialized: {webcam_config['width']}x{webcam_config['height']} @ {webcam_config['fps']}fps"
        )

    def _should_interact(self, detection: Detection) -> bool:
        """
        Check if we should interact with this person.

        Args:
            detection: The detected person

        Returns:
            True if we should interact (not in cooldown)
        """
        # Check minimum area
        min_area = self.config["detector"]["min_person_area"]
        if detection.area < min_area:
            return False

        # Check cooldown (if tracking is enabled)
        if detection.track_id is not None:
            last_time = self.last_interaction_time.get(detection.track_id, 0)
            if time.time() - last_time < self.cooldown_seconds:
                return False

        return True

    def _update_interaction_time(self, detection: Detection) -> None:
        """Record that we interacted with this person."""
        if detection.track_id is not None:
            self.last_interaction_time[detection.track_id] = time.time()

    def _capture_photo(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """
        Capture a cropped photo of the detected person.

        Args:
            frame: Full webcam frame
            detection: Detection with bounding box

        Returns:
            Cropped image of the person
        """
        # Add some padding around the detection
        padding = 50
        h, w = frame.shape[:2]

        x1 = max(0, detection.x1 - padding)
        y1 = max(0, detection.y1 - padding)
        x2 = min(w, detection.x2 + padding)
        y2 = min(h, detection.y2 + padding)

        return frame[y1:y2, x1:x2].copy()

    async def process_detection(
        self, frame: np.ndarray, detection: Detection
    ) -> None:
        """
        Process a detected person through the full pipeline.

        Args:
            frame: Full webcam frame
            detection: The detected person
        """
        start_time = time.time()
        logger.info(f"Processing detection: {detection}")

        try:
            # 1. Capture photo of the person
            photo = self._capture_photo(frame, detection)
            logger.debug(f"Photo captured: {photo.shape}")

            # 2. Get costume description from VLM
            costume_prompt = self.config["vlm"]["prompt"]
            description = self.vlm.describe_image(photo, costume_prompt)
            logger.info(f"Costume description: {description}")

            # 3. Generate response (different paths for voice lines vs real-time TTS)
            if self.use_voice_lines and self.voice_line_handler:
                audio_path = await self._process_with_voice_lines(description)
            else:
                audio_path = await self._process_with_realtime_tts(description)

            if audio_path:
                # 4. Play audio with lip-sync on avatar
                await self.avatar.play_audio_with_lipsync(audio_path)

            # Update interaction tracking
            self._update_interaction_time(detection)

            elapsed = time.time() - start_time
            logger.info(f"Interaction completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True)

    async def _process_with_voice_lines(self, costume_description: str) -> Optional[Path]:
        """
        Process using the voice line system with LLM tool-calling.

        Args:
            costume_description: Description from VLM

        Returns:
            Path to audio file, or None if failed
        """
        # Reset handler state
        self.voice_line_handler.reset()

        # Create user prompt for voice line selection
        user_message = create_costume_user_prompt(costume_description)

        # Use tool-calling to select voice line
        vl_config = self.config.get("voice_lines", {})

        if vl_config.get("use_tool_calling", True):
            # LLM selects the voice line via tools
            selected = await self._select_voice_line_with_tools(user_message)
        else:
            # Direct semantic search without LLM
            selected = self._select_voice_line_direct(costume_description)

        if not selected:
            logger.warning("No voice line selected, using fallback")
            selected = self.voice_line_handler.auto_select_best()

        if not selected or not selected.line:
            logger.error("Failed to select any voice line")
            return None

        logger.info(f"Selected voice line: {selected.line_id} ({selected.method})")
        logger.info(f"Voice line text: {selected.text}")

        # Get the audio file path
        audio_path = self.voice_line_tts._get_audio_path(selected.line)

        if not audio_path.exists():
            logger.error(f"Audio file missing: {audio_path}")
            # Try fallback to real-time TTS if configured
            if vl_config.get("fallback_to_tts", False) and self.tts:
                return await self._process_with_realtime_tts_text(selected.text)
            return None

        return audio_path

    async def _select_voice_line_with_tools(self, user_message: str) -> Optional[VoiceLineSelection]:
        """
        Use LLM tool-calling to select a voice line.

        Args:
            user_message: The user prompt with costume description

        Returns:
            Selected voice line, or None if failed
        """
        tools = self.voice_line_handler.get_tools()
        max_iterations = 3  # Prevent infinite tool-calling loops

        try:
            # Initial generation with tools
            result = self.llm.generate_with_tools(
                self.system_prompt,
                user_message,
                tools,
            )

            conversation_history = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result.text, "tool_calls": [
                    {"function": {"name": tc.name, "arguments": tc.arguments}, "id": tc.id}
                    for tc in result.tool_calls
                ] if result.tool_calls else None},
            ]

            iteration = 0
            while result.has_tool_calls and iteration < max_iterations:
                iteration += 1

                # Execute tool calls
                tool_results = []
                for tool_call in result.tool_calls:
                    tool_result = self.voice_line_handler.handle_tool_call(tool_call)
                    tool_results.append(tool_result)

                    logger.debug(f"Tool {tool_call.name}: {tool_result.content[:100]}...")

                # Check if a line was selected
                selected = self.voice_line_handler.get_selected_line()
                if selected:
                    return selected

                # Continue conversation with tool results
                result = self.llm.continue_with_tool_results(
                    self.system_prompt,
                    conversation_history,
                    tool_results,
                    tools,
                )

                # Update conversation history
                for tr in tool_results:
                    conversation_history.append({
                        "role": "tool",
                        "content": tr.content,
                        "tool_call_id": tr.tool_call_id,
                    })
                conversation_history.append({
                    "role": "assistant",
                    "content": result.text,
                })

            # Check final selection
            return self.voice_line_handler.get_selected_line()

        except Exception as e:
            logger.error(f"Tool-calling failed: {e}")
            return None

    def _select_voice_line_direct(self, costume_description: str) -> Optional[VoiceLineSelection]:
        """
        Select voice line using direct semantic search (no LLM).

        Args:
            costume_description: Description from VLM

        Returns:
            Selected voice line
        """
        # Extract costume type from description (simple keyword extraction)
        costume_type = self._extract_costume_type(costume_description)

        # Search for matching line
        line = self.voice_line_tts.get_best_line(
            desired_text=f"React to a {costume_type} costume",
            category_hint="costume_reactions",
            costume_type=costume_type,
        )

        if line:
            return VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method="direct_search",
            )

        return None

    def _extract_costume_type(self, description: str) -> str:
        """Extract costume type from VLM description."""
        # Simple keyword matching
        keywords = [
            "vampire", "zombie", "ghost", "skeleton", "witch", "wizard",
            "mummy", "werewolf", "frankenstein", "monster",
            "superhero", "hero", "ninja", "warrior", "knight",
            "princess", "queen", "king", "royalty",
            "fairy", "angel", "devil", "demon",
            "cat", "dog", "dinosaur", "shark", "animal",
            "pirate", "astronaut", "doctor", "nurse",
        ]

        description_lower = description.lower()
        for keyword in keywords:
            if keyword in description_lower:
                return keyword

        return "generic"

    async def _process_with_realtime_tts(self, costume_description: str) -> Optional[Path]:
        """
        Process using real-time TTS (legacy mode).

        Args:
            costume_description: Description from VLM

        Returns:
            Path to audio file
        """
        # Generate pirate response
        user_message = self.user_template.format(costume_description=costume_description)
        result = self.llm.generate(self.system_prompt, user_message)
        pirate_response = result.text
        logger.info(f"Pirate response: {pirate_response}")

        return await self._process_with_realtime_tts_text(pirate_response)

    async def _process_with_realtime_tts_text(self, text: str) -> Optional[Path]:
        """
        Synthesize text using real-time TTS.

        Args:
            text: Text to synthesize

        Returns:
            Path to audio file
        """
        audio_filename = f"response_{int(time.time())}.wav"
        audio_path = Path(self.config["tts"]["output_dir"]) / audio_filename

        self.tts.synthesize_to_file(text, audio_path)
        logger.info(f"Audio synthesized: {audio_path}")

        return audio_path

    async def run(self) -> None:
        """Main event loop."""
        self.running = True
        logger.info("PirateBot running! Press Ctrl+C to stop.")

        try:
            while self.running:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    await asyncio.sleep(0.1)
                    continue

                # Detect people
                # detections = self.detector.detect_people(frame)
                detections = []  # Placeholder until detector is implemented

                # Process each detection
                for detection in detections:
                    if self._should_interact(detection):
                        await self.process_detection(frame, detection)

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.033)  # ~30 FPS

        except asyncio.CancelledError:
            logger.info("Run loop cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Shutting down PirateBot...")
        self.running = False

        # Release webcam
        if self.cap is not None:
            self.cap.release()

        # Clean up components
        if self.detector:
            self.detector.cleanup()
        if self.vlm:
            self.vlm.cleanup()
        if self.tts:
            self.tts.cleanup()
        if self.voice_line_tts:
            self.voice_line_tts.cleanup()
        if self.avatar:
            await self.avatar.disconnect()

        logger.info("Shutdown complete")


async def main():
    """Entry point."""
    bot = PirateBot()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def signal_handler():
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.setup()
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
