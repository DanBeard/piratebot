#!/usr/bin/env python3
"""
Voice Line Expander - Generate more voice line variations using Ollama LLM.

Takes existing voice_lines.yaml and generates variations of each line
to create a larger, more diverse set of pirate responses.

Usage:
    python tools/expand_voice_lines.py                    # Expand all lines
    python tools/expand_voice_lines.py --variations 3    # 3 variations per line
    python tools/expand_voice_lines.py --category greetings  # Only expand category
    python tools/expand_voice_lines.py --dry-run         # Preview without saving
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EXPANSION_PROMPT = """You are a creative writer helping to expand a pirate voice line database for a Halloween decoration.

The pirate character is Captain Barnacle Bill - a friendly, theatrical pirate who greets trick-or-treaters.

Given an original pirate voice line, generate {num_variations} NEW variations that:
1. Keep the same general meaning and emotion
2. Use varied pirate vocabulary (Arrr, Ahoy, Blimey, Shiver me timbers, Avast, etc.)
3. Are family-friendly and enthusiastic
4. Are SHORT (1-2 sentences max)
5. Sound different from the original

Original line: "{original_text}"
Category: {category}
Subcategory: {subcategory}
Tags: {tags}
Emotion: {emotion}

Respond with ONLY a JSON array of strings, no other text:
["variation 1", "variation 2", ...]"""


class VoiceLineExpander:
    """Expands voice lines using Ollama LLM."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 60.0,
    ):
        """
        Initialize expander.

        Args:
            ollama_url: Ollama server URL
            model: Model to use for generation
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Statistics
        self.stats = {
            "lines_processed": 0,
            "variations_generated": 0,
            "errors": 0,
        }

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import httpx

            response = httpx.get(f"{self.ollama_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            # Check if our model is available
            model_base = self.model.split(":")[0]
            for m in models:
                if m.startswith(model_base):
                    return True

            logger.warning(f"Model {self.model} not found. Available: {models}")
            return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False

    def expand_line(
        self,
        text: str,
        category: str,
        subcategory: str,
        tags: list[str],
        emotion: str,
        num_variations: int = 2,
    ) -> list[str]:
        """
        Generate variations of a voice line.

        Args:
            text: Original line text
            category: Line category
            subcategory: Line subcategory
            tags: Line tags
            emotion: Line emotion
            num_variations: Number of variations to generate

        Returns:
            List of variation strings
        """
        import httpx

        prompt = EXPANSION_PROMPT.format(
            num_variations=num_variations,
            original_text=text,
            category=category,
            subcategory=subcategory,
            tags=", ".join(tags) if tags else "none",
            emotion=emotion,
        )

        try:
            response = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 500,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            response_text = data.get("response", "").strip()

            # Parse JSON array from response
            # Handle cases where model adds extra text
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning(f"Could not parse JSON array from response: {response_text[:100]}")
                self.stats["errors"] += 1
                return []

            json_str = response_text[start_idx:end_idx]
            variations = json.loads(json_str)

            if not isinstance(variations, list):
                logger.warning(f"Response is not a list: {type(variations)}")
                self.stats["errors"] += 1
                return []

            # Filter out empty or too-long variations
            valid_variations = [
                v.strip() for v in variations
                if isinstance(v, str) and v.strip() and len(v.strip()) < 200
            ]

            self.stats["variations_generated"] += len(valid_variations)
            return valid_variations

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            self.stats["errors"] += 1
            return []
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            self.stats["errors"] += 1
            return []
        except Exception as e:
            logger.error(f"Expansion failed: {e}")
            self.stats["errors"] += 1
            return []

    def load_yaml(self, yaml_path: str) -> dict:
        """Load voice lines YAML file."""
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def expand_all(
        self,
        yaml_path: str = "data/voice_lines.yaml",
        output_path: str = "data/voice_lines_expanded.yaml",
        num_variations: int = 2,
        category_filter: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Expand all voice lines and save to new file.

        Args:
            yaml_path: Path to original voice_lines.yaml
            output_path: Path to save expanded YAML
            num_variations: Variations per original line
            category_filter: Only expand this category
            dry_run: If True, don't save output

        Returns:
            Statistics dictionary
        """
        data = self.load_yaml(yaml_path)
        voice_lines = data.get("voice_lines", {})

        expanded_voice_lines = {}
        total_lines = 0
        expanded_idx = 0

        for category, subcategories in voice_lines.items():
            if category_filter and category != category_filter:
                # Copy without expansion
                expanded_voice_lines[category] = subcategories
                continue

            expanded_voice_lines[category] = {}

            for subcategory, entries in subcategories.items():
                expanded_entries = []

                for entry in entries:
                    total_lines += 1

                    # Parse entry
                    if isinstance(entry, str):
                        text = entry
                        tags = []
                        emotion = "neutral"
                        original_id = None
                    else:
                        text = entry["text"]
                        tags = entry.get("tags", [])
                        emotion = entry.get("emotion", "neutral")
                        original_id = entry.get("id")

                    # Add original entry
                    if isinstance(entry, dict):
                        expanded_entries.append(entry.copy())
                    else:
                        expanded_entries.append({
                            "id": f"{category}_{subcategory}_{expanded_idx:03d}",
                            "text": text,
                            "tags": tags,
                            "emotion": emotion,
                        })
                        expanded_idx += 1

                    # Generate variations
                    logger.info(f"Expanding: {text[:50]}...")

                    variations = self.expand_line(
                        text=text,
                        category=category,
                        subcategory=subcategory,
                        tags=tags,
                        emotion=emotion,
                        num_variations=num_variations,
                    )

                    self.stats["lines_processed"] += 1

                    # Add variations
                    for i, var_text in enumerate(variations):
                        base_id = original_id or f"{category}_{subcategory}"
                        var_id = f"{base_id}_var{i+1:02d}"

                        expanded_entries.append({
                            "id": var_id,
                            "text": var_text,
                            "tags": tags + ["variation"],
                            "emotion": emotion,
                        })

                        if not dry_run:
                            logger.debug(f"  Added variation: {var_text[:40]}...")

                expanded_voice_lines[category][subcategory] = expanded_entries

        # Prepare output data
        output_data = {
            "voice_lines": expanded_voice_lines,
            "metadata": {
                "version": "1.1-expanded",
                "source": yaml_path,
                "variations_per_line": num_variations,
                "total_lines": sum(
                    len(entries)
                    for subcats in expanded_voice_lines.values()
                    for entries in subcats.values()
                ),
                "generated_by": "expand_voice_lines.py",
            },
        }

        # Save if not dry run
        if not dry_run:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, width=120)

            logger.info(f"Saved expanded voice lines to: {output_path}")
        else:
            logger.info("Dry run - not saving output")

        # Print summary
        new_total = output_data["metadata"]["total_lines"]
        logger.info(
            f"Expansion complete: {total_lines} original -> {new_total} total "
            f"({self.stats['variations_generated']} variations generated, "
            f"{self.stats['errors']} errors)"
        )

        return self.stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Expand voice lines using Ollama LLM"
    )
    parser.add_argument(
        "--yaml",
        default="data/voice_lines.yaml",
        help="Path to input voice_lines.yaml",
    )
    parser.add_argument(
        "--output",
        default="data/voice_lines_expanded.yaml",
        help="Path for output expanded YAML",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=2,
        help="Number of variations per line",
    )
    parser.add_argument(
        "--category",
        help="Only expand this category",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without saving",
    )

    args = parser.parse_args()

    expander = VoiceLineExpander(
        ollama_url=args.ollama_url,
        model=args.model,
    )

    # Check Ollama availability
    if not expander.is_available():
        logger.error(
            f"Ollama is not available. Make sure it's running:\n"
            f"  ollama serve\n"
            f"  ollama pull {args.model}"
        )
        sys.exit(1)

    expander.expand_all(
        yaml_path=args.yaml,
        output_path=args.output,
        num_variations=args.variations,
        category_filter=args.category,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
