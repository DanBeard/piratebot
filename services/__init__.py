"""
Concrete implementations of PirateBot interfaces.
"""

from .yolo_detector import YoloDetector
from .moondream_vlm import MoondreamVLM
from .ollama_llm import OllamaLLM
from .kokoro_tts import KokoroTTS
from .godot_avatar import GodotAvatarController, MockAvatarController

__all__ = [
    "YoloDetector",
    "MoondreamVLM",
    "OllamaLLM",
    "KokoroTTS",
    "GodotAvatarController",
    "MockAvatarController",
]
