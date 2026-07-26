#!/usr/bin/env python3
"""
PirateBot - Interactive Halloween Pirate Decoration

Main orchestrator that coordinates all components:
- Webcam person detection
- Vision model for costume description
- Pre-generated voice line lookup via the cluster ``parrotts`` service
- 3D avatar control with lip-sync
"""

import argparse
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
    IAvatarController,
    Detection,
)

from services.appearance_cache import AppearanceCache
from services.interaction_state import InteractionIntent, InteractionManager
from services.parrotts_tts import ParrottsTTS, VoiceLineSelection

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
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.running = False

        # Components (will be initialized in setup())
        self.detector: Optional[IDetector] = None
        self.vlm: Optional[IVisionModel] = None
        self.llm: Optional[ILanguageModel] = None
        self.tts: Optional[ParrottsTTS] = None
        self.avatar: Optional[IAvatarController] = None
        self.props: Optional[Any] = None

        # Webcam
        self.cap: Optional[cv2.VideoCapture] = None

        # Idle state
        self._idle_task: Optional[asyncio.Task] = None
        self._last_any_interaction_time = 0.0

        # State tracking
        self.last_interaction_time: dict[int, float] = {}  # track_id -> timestamp
        self.cooldown_seconds = self.config["detector"]["cooldown_seconds"]

        # Appearance cache for cross-track re-identification
        tracking_config = self.config.get("tracking", {})
        self.appearance_cache = AppearanceCache(
            ttl_seconds=tracking_config.get("appearance_cache_ttl", 600),
            similarity_threshold=tracking_config.get("appearance_similarity", 0.85),
        )

        # Interaction state machine
        self.interaction_manager = InteractionManager()

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

        from services.yolo_detector import YoloDetector
        from services.moondream_vlm import MoondreamVLM
        from services.ollama_llm import OllamaLLM
        from services.godot_avatar import GodotAvatarController

        gpu_vlm = self.config["gpu"]["vlm"]
        vlm_device = (
            f"cuda:{gpu_vlm}"
            if isinstance(gpu_vlm, int)
            else gpu_vlm
        )

        self.detector = YoloDetector(
            model=self.config["detector"]["model"],
            confidence_threshold=self.config["detector"]["confidence_threshold"],
            device=vlm_device,
        )
        self.detector.warmup()

        self.vlm = MoondreamVLM(
            model_id=self.config["vlm"]["model_id"],
            revision=self.config["vlm"].get("revision"),
            device=vlm_device,
        )
        self.vlm.warmup()

        # LLM is optional. When enabled, it must be reachable via Ollama.
        # The default porch path uses the pre-generated parrotts voice library,
        # so no live LLM generation is required.
        llm_config = self.config.get("llm", {})
        if llm_config.get("enabled", False):
            self.llm = OllamaLLM(
                base_url=llm_config["base_url"],
                model=llm_config["model"],
            )
            self.llm.warmup()

        await self._setup_parrotts()
        await self._setup_props()

        # Choose avatar backend: portrait (browser-based 2D) or Godot 3D.
        portrait_config = self.config.get("portrait")
        if portrait_config:
            from services.portrait_avatar import PortraitAvatarController
            self.avatar = PortraitAvatarController(
                host=portrait_config.get("host", "localhost"),
                port=portrait_config.get("port", 9877),
                assets_dir=portrait_config.get("assets_dir", "portrait_viewer"),
                visemes_dir=portrait_config.get("visemes_dir", "data/parrotts_cache"),
                asset_set=portrait_config.get("asset_set", "default"),
            )
            await self.avatar.connect()
        else:
            from services.godot_avatar import GodotAvatarController
            self.avatar = GodotAvatarController(
                host=self.config["avatar"]["host"],
                port=self.config["avatar"]["port"],
            )
            await self.avatar.connect()

        self._setup_webcam()

        logger.info("All components initialized")

    async def _setup_parrotts(self) -> None:
        """Initialize the parrotts cluster TTS backend."""
        pt_config = self.config["parrotts"]
        logger.info(
            "Setting up parrotts TTS (base_url=%s character=%s)",
            pt_config["base_url"],
            pt_config["character"],
        )
        self.tts = ParrottsTTS(
            base_url=pt_config["base_url"],
            character=pt_config["character"],
            cache_dir=pt_config.get("cache_dir", "data/parrotts_cache"),
        )
        self.tts.warmup()
        logger.info("Parrotts TTS ready")

    async def _setup_props(self) -> None:
        """Initialize the prop mesh if enabled."""
        pm_config = self.config.get("prop_mesh", {})
        if not pm_config.get("enabled", False):
            logger.info("Prop mesh disabled")
            return

        from services.prop_mesh import PropMeshBus
        from services.prop_controller import PropController

        self.props = PropController(
            mesh=PropMeshBus(
                source=pm_config.get("source", "piratebot"),
                mode=pm_config.get("mode", "server"),
                host=pm_config.get("host", "0.0.0.0"),
                port=pm_config.get("port", 9001),
                broker_url=pm_config.get("broker_url"),
            ),
            avatar=self.avatar,
            auto_triggers=pm_config.get("auto_triggers", {}),
        )
        await self.props.mesh.connect()
        logger.info("Prop mesh ready")

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

            # 2.5. Determine interaction intent based on appearance cache
            if self.appearance_cache.is_known(description):
                intent = InteractionIntent.RETURNING
                logger.info("Returning visitor detected, selecting returning greeting")
            else:
                intent = InteractionIntent.COSTUME_REACT

            # 3. Select voice line based on intent
            selection = await self._select_with_intent(intent, description)
            if selection:
                await self._play_selection(selection)

            # Cache appearance and mark interacted on success
            if intent == InteractionIntent.COSTUME_REACT:
                self.appearance_cache.add(description)

            if detection.track_id is not None:
                self.interaction_manager.mark_interacted(
                    detection.track_id, description
                )

            # Update interaction tracking
            self._update_interaction_time(detection)

            elapsed = time.time() - start_time
            logger.info(f"Interaction completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Error processing detection: {e}", exc_info=True)

    async def _select_with_intent(
        self, intent: InteractionIntent, costume_description: str
    ) -> Optional[VoiceLineSelection]:
        """Select a voice line for the current intent."""
        if intent == InteractionIntent.RETURNING:
            return self._select_voice_line_for_intent(intent)
        return self._select_voice_line_direct(costume_description)

    async def _play_selection(self, selection: VoiceLineSelection) -> None:
        """Resolve audio, trigger props, and play the selected line."""
        audio_path = self.tts._get_audio_path(selection.line)

        # Notify prop mesh before speaking so effects sync with audio start.
        if self.props:
            await self.props.on_speak(
                line_id=selection.line_id,
                text=selection.line.text,
                emotion=selection.line.emotion,
                tags=selection.line.tags,
            )

        await self.avatar.play_audio_with_lipsync(audio_path)
        self._last_any_interaction_time = time.time()

    async def _process_with_intent(
        self, intent: InteractionIntent, costume_description: str
    ) -> Optional[Path]:
        """Legacy helper: resolve selected line to audio path."""
        selection = await self._select_with_intent(intent, costume_description)
        if selection is None:
            return None
        return self.tts._get_audio_path(selection.line)

    async def _play_farewell(self, state) -> None:
        """Play a farewell voice line for a departing person."""
        line = self.tts.get_db().get_random_from_category("farewells")
        if not line:
            logger.debug("No farewell lines available")
            return

        try:
            audio_path = self.tts._get_audio_path(line)
        except Exception as e:
            logger.warning(f"Farewell fetch failed for {line.id}: {e}")
            return

        logger.info(f"Playing farewell for track {state.track_id}: {line.text}")

        if self.props:
            await self.props.on_speak(
                line_id=line.id,
                text=line.text,
                emotion=line.emotion,
                tags=line.tags,
            )

        await self.avatar.play_audio_with_lipsync(audio_path)
        self._last_any_interaction_time = time.time()

    def _select_voice_line_for_intent(
        self, intent: InteractionIntent
    ) -> Optional[VoiceLineSelection]:
        """Select a voice line for non-costume-react intents."""
        db = self.tts.get_db()

        if intent == InteractionIntent.RETURNING:
            line = db.get_random_from_category("greetings", "returning")
        elif intent == InteractionIntent.FAREWELL:
            line = db.get_random_from_category("farewells")
        else:
            return None

        if line:
            return VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method=f"random_{intent.value}",
            )
        return None

    def _select_voice_line_direct(self, costume_description: str) -> Optional[VoiceLineSelection]:
        """Select voice line using direct semantic search against parrotts."""
        costume_type = self._extract_costume_type(costume_description)

        line = self.tts.get_best_line(
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
        """Extract costume type from VLM description via keyword scan."""
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
                detections = self.detector.detect_people(frame)

                # Get track lifecycle events from the most recent detect()
                arrived, departed = self.detector.get_track_events()

                # 1. Departures → farewell (no VLM, just play a farewell line)
                for track_id in departed:
                    state = self.interaction_manager.on_departure(track_id)
                    if state and state.interacted:
                        await self._play_farewell(state)
                    if self.props:
                        await self.props.on_departure(track_id)

                # 2. Arrivals → register in state manager
                for track_id in arrived:
                    self.interaction_manager.on_arrival(track_id)
                    if self.props:
                        await self.props.on_arrival(track_id)

                # 3. Process detections (existing _should_interact filter)
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

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Clean up components
        if self.detector:
            self.detector.cleanup()
            self.detector = None
        if self.vlm:
            self.vlm.cleanup()
            self.vlm = None
        if self.llm:
            self.llm = None
        if self.tts:
            self.tts.cleanup()
            self.tts = None
        if self.avatar:
            try:
                await self.avatar.disconnect()
            except Exception as e:
                logger.warning(f"Avatar disconnect error: {e}")
            self.avatar = None
        if self.props:
            try:
                await self.props.mesh.disconnect()
            except Exception as e:
                logger.warning(f"Prop mesh disconnect error: {e}")
            self.props = None

        logger.info("Shutdown complete")


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="PirateBot Halloween pirate")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    bot = PirateBot(config_path=args.config)

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
