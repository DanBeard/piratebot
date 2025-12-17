"""
Abstract interfaces for PirateBot components.
All services implement these interfaces to enable easy swapping of implementations.
"""

from .detector import IDetector, Detection
from .vision_model import IVisionModel
from .language_model import ILanguageModel
from .tts_engine import ITTSEngine
from .avatar_controller import IAvatarController

__all__ = [
    "IDetector",
    "Detection",
    "IVisionModel",
    "ILanguageModel",
    "ITTSEngine",
    "IAvatarController",
]
