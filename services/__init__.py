"""
Concrete implementations of PirateBot interfaces.
"""

from .yolo_detector import YoloDetector
from .moondream_vlm import MoondreamVLM
from .ollama_llm import OllamaLLM, ToolDefinition, ToolCall, ToolResult, GenerationWithToolsResult
from .kokoro_tts import KokoroTTS
from .godot_avatar import GodotAvatarController, MockAvatarController
from .voice_line_db import VoiceLineDB, VoiceLine, SearchResult
from .voice_line_tts import VoiceLineTTS
from .voice_line_search import (
    VoiceLineToolHandler,
    VoiceLineSelection,
    VOICE_LINE_TOOLS,
    get_voice_line_system_prompt,
    create_costume_user_prompt,
)

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
    # TTS - Legacy
    "KokoroTTS",
    # TTS - Voice Lines
    "VoiceLineDB",
    "VoiceLine",
    "SearchResult",
    "VoiceLineTTS",
    "VoiceLineToolHandler",
    "VoiceLineSelection",
    "VOICE_LINE_TOOLS",
    "get_voice_line_system_prompt",
    "create_costume_user_prompt",
    # Avatar
    "GodotAvatarController",
    "MockAvatarController",
]
