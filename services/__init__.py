"""
Concrete implementations of PirateBot interfaces.
"""

from .yolo_detector import YoloDetector
from .moondream_vlm import MoondreamVLM
from .ollama_llm import OllamaLLM, ToolDefinition, ToolCall, ToolResult, GenerationWithToolsResult
from .godot_avatar import GodotAvatarController, MockAvatarController
from .parrotts_tts import ParrottsTTS, VoiceLine, SearchResult, VoiceLineSelection

__all__ = [
    # Detector
    "YoloDetector",
    # Vision
    "MoondreamVLM",
    # Language Model
    "OllamaLLM",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "GenerationWithToolsResult",
    # TTS — parrotts cluster service
    "ParrottsTTS",
    "VoiceLine",
    "SearchResult",
    "VoiceLineSelection",
    # Avatar
    "GodotAvatarController",
    "MockAvatarController",
]
