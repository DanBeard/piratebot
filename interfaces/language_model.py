"""
Interface for Language Models (LLM).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result from text generation."""

    text: str
    tokens_generated: int = 0
    generation_time_ms: float = 0.0
    model: str = ""


class ILanguageModel(ABC):
    """
    Abstract interface for Language Models.

    Implementations:
    - OllamaLLM: Ollama-hosted models (Llama, Mistral, etc.)
    - (Future) TransformersLLM: HuggingFace Transformers models
    - (Future) OpenAILLM: OpenAI API (for fallback/comparison)
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response given a system prompt and user message.

        Args:
            system_prompt: The system/persona prompt (e.g., pirate character)
            user_message: The user's message to respond to
            config: Optional generation parameters

        Returns:
            GenerationResult containing the generated text
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Async version of generate().
        """
        pass

    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Optional: Stream generated tokens as they're produced.

        Yields:
            Individual tokens or chunks of text
        """
        # Default implementation: just yield the full result
        result = await self.generate_async(system_prompt, user_message, config)
        yield result.text

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available and ready."""
        pass

    def warmup(self) -> None:
        """
        Optional: Send a warmup request to load model into memory.
        """
        pass

    def get_model_info(self) -> dict:
        """
        Optional: Return information about the configured model.

        Returns:
            Dict with keys like 'name', 'parameters', 'context_length', etc.
        """
        return {}
