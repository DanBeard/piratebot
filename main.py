#!/usr/bin/env python3
"""
PirateBot - Interactive Halloween Pirate Decoration

Main orchestrator that coordinates all components:
- Webcam person detection
- Vision model for costume description
- LLM for pirate banter generation
- TTS for voice synthesis
- 3D avatar control with lip-sync
"""

import asyncio
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

        # Webcam
        self.cap: Optional[cv2.VideoCapture] = None

        # State tracking
        self.last_interaction_time: dict[int, float] = {}  # track_id -> timestamp
        self.cooldown_seconds = self.config["detector"]["cooldown_seconds"]

        # Load prompts
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
        # (These will be uncommented as implementations are built)

        # from services.yolo_detector import YoloDetector
        # from services.moondream_vlm import MoondreamVLM
        # from services.ollama_llm import OllamaLLM
        # from services.kokoro_tts import KokoroTTS
        # from services.godot_avatar import GodotAvatarController

        # Initialize detector
        # self.detector = YoloDetector(
        #     model=self.config["detector"]["model"],
        #     confidence_threshold=self.config["detector"]["confidence_threshold"],
        # )
        # self.detector.warmup()

        # Initialize VLM
        # self.vlm = MoondreamVLM(
        #     model_id=self.config["vlm"]["model_id"],
        #     device=f"cuda:{self.config['gpu']['vlm']}",
        # )
        # self.vlm.warmup()

        # Initialize LLM
        # self.llm = OllamaLLM(
        #     base_url=self.config["llm"]["base_url"],
        #     model=self.config["llm"]["model"],
        # )
        # self.llm.warmup()

        # Initialize TTS
        # self.tts = KokoroTTS(
        #     voice=self.config["tts"]["voice"],
        #     output_dir=self.config["tts"]["output_dir"],
        # )
        # self.tts.warmup()

        # Initialize Avatar
        # self.avatar = GodotAvatarController(
        #     host=self.config["avatar"]["host"],
        #     port=self.config["avatar"]["port"],
        # )
        # await self.avatar.connect()

        # Initialize webcam
        self._setup_webcam()

        logger.info("All components initialized")

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
            # description = self.vlm.describe_image(photo, costume_prompt)
            description = "[VLM not yet implemented] A person in a Halloween costume"
            logger.info(f"Costume description: {description}")

            # 3. Generate pirate response
            user_message = self.user_template.format(costume_description=description)
            # result = self.llm.generate(self.system_prompt, user_message)
            # pirate_response = result.text
            pirate_response = "Arrr! Ahoy there, little matey! That be a fine costume ye got!"
            logger.info(f"Pirate response: {pirate_response}")

            # 4. Synthesize speech
            audio_filename = f"response_{int(time.time())}.wav"
            audio_path = Path(self.config["tts"]["output_dir"]) / audio_filename
            # self.tts.synthesize_to_file(pirate_response, audio_path)
            logger.info(f"Audio synthesized: {audio_path}")

            # 5. Play audio with lip-sync on avatar
            # await self.avatar.play_audio_with_lipsync(audio_path)

            # Update interaction tracking
            self._update_interaction_time(detection)

            elapsed = time.time() - start_time
            logger.info(f"Interaction completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True)

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
