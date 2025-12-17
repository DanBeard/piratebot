"""
Interface for Vision Language Models (VLM).
"""

from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image


class IVisionModel(ABC):
    """
    Abstract interface for Vision Language Models.

    Implementations:
    - MoondreamVLM: Moondream2 model (1.86B params, fast)
    - (Future) LlavaVLM: LLaVA model (7B+ params, more capable)
    - (Future) SmolVLM: SmolVLM model (2B params)
    """

    @abstractmethod
    def describe_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        prompt: str,
    ) -> str:
        """
        Generate a text description of the image based on the prompt.

        Args:
            image: Input image as numpy array (BGR), PIL Image, or path to image file
            prompt: The question or instruction for the VLM
                   (e.g., "Describe this person's Halloween costume")

        Returns:
            Generated text description
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        pass

    def warmup(self) -> None:
        """
        Optional: Run a warmup inference to load model into GPU memory.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional: Clean up resources and free GPU memory.
        """
        pass

    def get_model_info(self) -> dict:
        """
        Optional: Return information about the loaded model.

        Returns:
            Dict with keys like 'name', 'parameters', 'device', etc.
        """
        return {}
