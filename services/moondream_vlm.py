"""
Moondream2 Vision Language Model implementation.

Moondream2 is a small (1.86B params) but capable VLM optimized for
fast inference on consumer hardware.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from interfaces.vision_model import IVisionModel

logger = logging.getLogger(__name__)


class MoondreamVLM(IVisionModel):
    """
    Vision Language Model using Moondream2.

    Moondream2 is a 1.86B parameter model that's optimized for
    edge/on-device inference while maintaining good accuracy.
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        """
        Initialize Moondream2 VLM.

        Args:
            model_id: HuggingFace model ID
            revision: Specific model revision (recommended for stability)
            device: Device to run on ("cuda", "cuda:0", "cpu")
            torch_dtype: Precision ("float16", "float32", "bfloat16")
        """
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.torch_dtype_str = torch_dtype

        self._model = None
        self._tokenizer = None
        self._loaded = False

        logger.info(
            f"MoondreamVLM initialized: model={model_id}, device={device}"
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Map string dtype to torch dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)

            logger.info(f"Loading Moondream2 model: {self.model_id}")

            # Load model with trust_remote_code since Moondream uses custom code
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map={"": self.device},
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
            )

            self._loaded = True
            logger.info("Moondream2 model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Required package not installed: {e}. "
                "Install with: pip install transformers torch"
            )

    def _prepare_image(
        self, image: Union[np.ndarray, Image.Image, str, Path]
    ) -> Image.Image:
        """
        Convert input to PIL Image.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            # Assume BGR (OpenCV format), convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            return Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def describe_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        prompt: str,
    ) -> str:
        """
        Generate a text description of the image.

        Args:
            image: Input image
            prompt: Question or instruction for the VLM

        Returns:
            Generated text description
        """
        self._ensure_loaded()

        start_time = time.time()

        # Prepare image
        pil_image = self._prepare_image(image)

        # Moondream2 uses a specific format for image+text queries
        # The model has an encode_image method and an answer method
        try:
            # Encode the image
            enc_image = self._model.encode_image(pil_image)

            # Generate answer
            answer = self._model.answer_question(enc_image, prompt, self._tokenizer)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"VLM inference completed in {elapsed_ms:.0f}ms")

            return answer.strip()

        except AttributeError:
            # Fallback for different Moondream2 API versions
            logger.warning("Using fallback generation method")
            return self._fallback_generate(pil_image, prompt)

    def _fallback_generate(
        self, image: Image.Image, prompt: str
    ) -> str:
        """
        Fallback generation method for different API versions.
        """
        import torch

        # Prepare inputs
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Some versions might use different generation methods
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response if it's included
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def warmup(self) -> None:
        """
        Run warmup inference to load model into GPU memory.
        """
        self._ensure_loaded()

        logger.info("Running Moondream2 warmup inference...")

        # Create a small dummy image for warmup
        dummy_image = Image.new("RGB", (224, 224), color="gray")

        try:
            _ = self.describe_image(dummy_image, "What is in this image?")
            logger.info("Moondream2 warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def cleanup(self) -> None:
        """
        Clean up model resources.
        """
        if self._model is not None:
            try:
                import torch

                del self._model
                del self._tokenizer

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

            self._model = None
            self._tokenizer = None
            self._loaded = False

            logger.info("Moondream2 model unloaded")

    def get_model_info(self) -> dict:
        """Return information about the model."""
        return {
            "name": "Moondream2",
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.torch_dtype_str,
            "parameters": "1.86B",
            "loaded": self._loaded,
        }


# Convenience function for testing
def test_costume_description():
    """Test Moondream2 with a sample image."""
    import sys

    # Check for image path argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python moondream_vlm.py <image_path>")
        print("Testing with dummy image...")
        image_path = None

    vlm = MoondreamVLM()
    vlm.warmup()

    prompt = "Describe this person's Halloween costume in one brief sentence. Focus on what character or thing they are dressed as."

    if image_path:
        print(f"Analyzing image: {image_path}")
        description = vlm.describe_image(image_path, prompt)
    else:
        # Test with a simple colored image
        test_image = Image.new("RGB", (512, 512), color="orange")
        print("Analyzing test image (orange square)...")
        description = vlm.describe_image(test_image, "What color is this image?")

    print(f"\nDescription: {description}")

    # Print model info
    info = vlm.get_model_info()
    print(f"\nModel info: {info}")

    vlm.cleanup()


if __name__ == "__main__":
    test_costume_description()
