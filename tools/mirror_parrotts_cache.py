#!/usr/bin/env python3
"""Mirror the cluster parrotts voice library to the local porch PC.

This downloads every line from ``data/voice_lines.yaml`` from the cluster
parrotts service into ``data/parrotts_cache/``.  Once mirrored, the porch PC
can run PirateBot even if the cluster / internet goes down.

Usage:

    python tools/mirror_parrotts_cache.py --base-url http://localhost:18003

The default base URL assumes a kubectl port-forward:

    kubectl port-forward -n default svc/parrotts 18003:8000

Re-running is safe and fast — existing files are skipped unless --force is
passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.parrotts_vendor import ParrottsClient, ParrottsError  # noqa: E402


def collect_lines_from_yaml(yaml_path: Path) -> list[dict[str, Any]]:
    """Flatten voice_lines.yaml into the parrotts batch payload shape."""
    data = yaml.safe_load(yaml_path.read_text())
    out: list[dict[str, Any]] = []
    for category, subcategories in data.get("voice_lines", {}).items():
        for subcategory, entries in subcategories.items():
            for entry in entries:
                if isinstance(entry, str):
                    out.append({
                        "text": entry,
                        "category": category,
                        "subcategory": subcategory,
                    })
                    continue
                line: dict[str, Any] = {
                    "text": entry["text"],
                    "category": category,
                    "subcategory": subcategory,
                }
                if entry.get("id"):
                    line["id"] = entry["id"]
                if entry.get("tags"):
                    line["tags"] = list(entry["tags"])
                if entry.get("emotion"):
                    line["emotion"] = entry["emotion"]
                out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:18003",
        help="parrotts base URL (default: http://localhost:18003)",
    )
    parser.add_argument("--character", default="pirate")
    parser.add_argument(
        "--yaml",
        default=str(REPO_ROOT / "data" / "voice_lines.yaml"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "data" / "parrotts_cache"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    lines = collect_lines_from_yaml(yaml_path)
    print(f"Loaded {len(lines)} lines from {yaml_path}")

    print(f"Connecting to parrotts at {args.base_url}")
    client = ParrottsClient(base_url=args.base_url, timeout=args.timeout)
    try:
        info = client._http.get("/v1/info").json()
        print(
            f"  parrotts {info.get('version')} — "
            f"{info.get('line_count', '?')} lines already in library"
        )
    except Exception as e:
        print(f"ERROR: cannot reach parrotts at {args.base_url}: {e}")
        return 1

    missing = 0
    cached = 0
    downloaded = 0
    failed = 0

    for line in lines:
        line_id = line.get("id")
        if not line_id:
            print(f"  SKIP: no id for line {line['text'][:40]!r}")
            continue

        dest = cache_dir / f"{line_id}.wav"
        if dest.exists() and not args.force:
            cached += 1
            continue

        try:
            client.download_line(line_id, dest)
            downloaded += 1
            print(f"  OK: {line_id} -> {dest.name}")
        except ParrottsError as e:
            failed += 1
            print(f"  FAIL: {line_id}: {e}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {line_id}: {e}")

    print()
    print(f"Mirror complete: downloaded={downloaded} cached={cached} failed={failed}")

    if failed > 0:
        print("Some lines failed. Re-run the script to retry.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
