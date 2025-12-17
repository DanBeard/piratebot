"""
Ollama LLM implementation.

Uses the Ollama REST API to generate text with local LLMs.
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

import aiohttp
import httpx

from interfaces.language_model import (
    ILanguageModel,
    GenerationConfig,
    GenerationResult,
)

logger = logging.getLogger(__name__)


class OllamaLLM(ILanguageModel):
    """
    Language model implementation using Ollama.

    Ollama provides a simple REST API for running local LLMs like
    Llama, Mistral, Phi, etc.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama LLM client.

        Args:
            base_url: Ollama server URL
            model: Model name (e.g., "llama3.2:3b", "mistral", "phi3")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Sync client for non-async usage
        self._sync_client = httpx.Client(timeout=timeout)

        logger.info(f"OllamaLLM initialized: model={model}, url={base_url}")

    def _build_messages(
        self, system_prompt: str, user_message: str
    ) -> list[dict]:
        """Build chat messages in Ollama format."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _build_request(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> dict:
        """Build the Ollama API request body."""
        config = config or GenerationConfig()

        request = {
            "model": self.model,
            "messages": self._build_messages(system_prompt, user_message),
            "stream": stream,
            "options": {
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }

        if config.stop_sequences:
            request["options"]["stop"] = config.stop_sequences

        return request

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response synchronously.

        Args:
            system_prompt: System/persona prompt
            user_message: User's message
            config: Generation parameters

        Returns:
            GenerationResult with generated text
        """
        start_time = time.time()

        request_body = self._build_request(
            system_prompt, user_message, config, stream=False
        )

        try:
            response = self._sync_client.post(
                f"{self.base_url}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            text = data["message"]["content"]
            tokens = data.get("eval_count", 0)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Generated {tokens} tokens in {elapsed_ms:.0f}ms"
            )

            return GenerationResult(
                text=text,
                tokens_generated=tokens,
                generation_time_ms=elapsed_ms,
                model=self.model,
            )

        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def generate_async(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response asynchronously.

        Args:
            system_prompt: System/persona prompt
            user_message: User's message
            config: Generation parameters

        Returns:
            GenerationResult with generated text
        """
        start_time = time.time()

        request_body = self._build_request(
            system_prompt, user_message, config, stream=False
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=request_body,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    text = data["message"]["content"]
                    tokens = data.get("eval_count", 0)
                    elapsed_ms = (time.time() - start_time) * 1000

                    logger.debug(
                        f"Generated {tokens} tokens in {elapsed_ms:.0f}ms"
                    )

                    return GenerationResult(
                        text=text,
                        tokens_generated=tokens,
                        generation_time_ms=elapsed_ms,
                        model=self.model,
                    )

            except aiohttp.ClientError as e:
                logger.error(f"Ollama API error: {e}")
                raise

    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens as they're produced.

        Yields:
            Individual tokens or chunks of text
        """
        request_body = self._build_request(
            system_prompt, user_message, config, stream=True
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=request_body,
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            response = self._sync_client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            # Check if our model is available (handle tags like :latest)
            model_base = self.model.split(":")[0]
            for m in models:
                if m.startswith(model_base):
                    return True

            logger.warning(
                f"Model {self.model} not found. Available: {models}"
            )
            return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False

    def warmup(self) -> None:
        """
        Send a warmup request to load model into GPU memory.
        """
        logger.info(f"Warming up Ollama model: {self.model}")

        try:
            # Send a short generation to load the model
            result = self.generate(
                system_prompt="You are a helpful assistant.",
                user_message="Say hello in one word.",
                config=GenerationConfig(max_tokens=10),
            )
            logger.info(f"Warmup complete: {result.text[:50]}...")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def get_model_info(self) -> dict:
        """Return information about the configured model."""
        try:
            response = self._sync_client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

        return {
            "name": self.model,
            "base_url": self.base_url,
        }

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_sync_client"):
            self._sync_client.close()


# Convenience function for testing
def test_pirate_response():
    """Test the Ollama LLM with a pirate prompt."""
    llm = OllamaLLM()

    if not llm.is_available():
        print("Ollama is not available. Make sure it's running:")
        print("  ollama serve")
        print(f"  ollama pull {llm.model}")
        return

    system_prompt = """You are Captain Barnacle Bill, a friendly pirate who loves Halloween!
You speak with a pirate accent using words like "Arrr", "matey", "ahoy",
"shiver me timbers", and "blimey".

Keep responses SHORT (1-2 sentences max). Be enthusiastic and encouraging!"""

    user_message = """A trick-or-treater just arrived! Here's what I see: A small child dressed as a vampire with a black cape and plastic fangs.

Give them a short, fun pirate comment about their costume!"""

    print("Testing pirate response...")
    print("-" * 50)

    result = llm.generate(system_prompt, user_message)

    print(f"Response: {result.text}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Time: {result.generation_time_ms:.0f}ms")


if __name__ == "__main__":
    test_pirate_response()
