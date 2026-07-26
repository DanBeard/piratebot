#!/usr/bin/env python3
"""Draft new voice lines with an LLM for human curation.

This tool is the first stage of growing the pirate voice corpus safely:
1. It asks a local LLM (via litellm) for line suggestions on a theme.
2. It writes them to a draft YAML file with `approved: false`.
3. A human reviews, edits, and flips `approved: true`.
4. `tools/migrate_to_parrotts.py` only migrates approved lines.

No line produced by this tool can ever reach the live show until it is
explicitly approved and voice-generated.

Usage:

    python tools/expand_voice_lines.py \
        --theme "comebacks when kids say the pirate is fake" \
        --count 20 \
        --category comebacks \
        --subcategory fake \
        --emotion grumpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROMPT = """You are Captain Barnacle Bill, a friendly but grumpy skeleton pirate who loves Halloween.
You speak in theatrical pirate dialect: "ye", "yer", "me", "avast", "blimey", "shiver me timbers", "landlubber", "scallywag", etc.
You are family-friendly. Never use modern slang, profanity, insults about real groups, or anything mean about a child's appearance.
Keep each line SHORT: 1 sentence, ideally 6-12 words.
Return exactly the requested number of lines, each on its own line, no numbering, no quotes.

Theme: {theme}"""


def load_existing_ids(yaml_path: Path) -> set[str]:
    if not yaml_path.exists():
        return set()
    data = yaml.safe_load(yaml_path.read_text()) or {}
    ids: set[str] = set()
    for category, subcategories in data.get("voice_lines", {}).items():
        for subcategory, entries in subcategories.items():
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    ids.add(entry["id"])
    return ids


def next_id(prefix: str, existing: set[str]) -> str:
    """Find the next available zero-padded id, e.g. fake_001."""
    for n in range(1, 9999):
        candidate = f"{prefix}_{n:03d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("id space exhausted")


def call_llm(base_url: str, model: str, prompt: str, count: int, api_key: str = "none") -> list[str]:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Give me {count} short pirate lines on this theme."},
    ]

    headers = {"Authorization": f"Bearer {api_key}"} if api_key and api_key != "none" else {}

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": count * 40,
                "temperature": 0.85,
                "top_p": 0.9,
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

    # Parse numbered or plain lines.
    lines: list[str] = []
    for raw in content.strip().split("\n"):
        line = raw.strip().strip('"').strip("-")
        if not line:
            continue
        # Remove leading numbering like "1." or "1)"
        import re
        line = re.sub(r"^\d+[\.\)\-]\s*", "", line)
        if line:
            lines.append(line)
    return lines[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", required=True, help="What kind of lines to draft")
    parser.add_argument("--count", type=int, default=10, help="Lines to draft")
    parser.add_argument("--category", required=True, help="voice_lines.yaml category")
    parser.add_argument("--subcategory", required=True, help="voice_lines.yaml subcategory")
    parser.add_argument("--emotion", default="neutral", help="Emotion tag for these lines")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument(
        "--base-url",
        default="http://192.168.0.14:4000/v1",
        help="litellm / OpenAI-compatible endpoint",
    )
    parser.add_argument("--model", default="local-moderate")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    parser.add_argument(
        "--yaml",
        default=str(REPO_ROOT / "data" / "voice_lines.yaml"),
    )
    parser.add_argument(
        "--draft-yaml",
        help="Write drafts to a separate file instead of the main voice_lines.yaml",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    existing_ids = load_existing_ids(yaml_path)

    prefix = f"{args.category}_{args.subcategory}".lower().replace("/", "_")
    if len(prefix) > 40:
        prefix = prefix[:40]

    print(f"Drafting {args.count} lines for {args.category}/{args.subcategory}...")
    print(f"LLM: {args.base_url} model={args.model}")

    prompt = DEFAULT_PROMPT.format(theme=args.theme)
    lines = call_llm(args.base_url, args.model, prompt, args.count, args.api_key)

    if not lines:
        print("ERROR: LLM returned no lines")
        return 1

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    new_entries: list[dict[str, Any]] = []
    for text in lines:
        line_id = next_id(prefix, existing_ids)
        existing_ids.add(line_id)
        new_entries.append({
            "id": line_id,
            "text": text,
            "tags": tags,
            "emotion": args.emotion,
            "approved": False,
        })
        print(f"  DRAFT {line_id}: {text}")

    # Write to main YAML or a draft file.
    target_path = Path(args.draft_yaml) if args.draft_yaml else yaml_path
    if target_path.exists():
        data = yaml.safe_load(target_path.read_text()) or {}
    else:
        data = {"voice_lines": {}}

    voice_lines = data.setdefault("voice_lines", {})
    category = voice_lines.setdefault(args.category, {})
    subcategory = category.setdefault(args.subcategory, [])
    subcategory.extend(new_entries)

    # Add metadata if missing.
    if "metadata" not in data:
        data["metadata"] = {"version": "2.1", "categories": []}
    if args.category not in data["metadata"].get("categories", []):
        data["metadata"].setdefault("categories", []).append(args.category)

    target_path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))
    print(f"\nWrote {len(new_entries)} draft(s) to {target_path}")
    print("Review them, edit if needed, then set approved: true before migrating to parrotts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
