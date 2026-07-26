#!/usr/bin/env python3
"""Pre-compute Rhubarb visemes for every WAV in the parrotts cache.

This makes portrait-mode lip sync fast at showtime: the browser loads a
small JSON file alongside each audio file instead of running Rhubarb live.

Usage:

    python tools/batch_rhubarb.py --cache-dir data/parrotts_cache
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_rhubarb(wav_path: Path) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["rhubarb", str(wav_path), "-f", "json", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Rhubarb not found in PATH. Install from "
            "https://github.com/DanielSWolf/rhubarb-lip-sync"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Rhubarb timed out on {wav_path.name}") from exc

    if result.returncode != 0:
        raise RuntimeError(f"Rhubarb failed on {wav_path.name}: {result.stderr}")

    data = json.loads(result.stdout)
    return data.get("mouthCues", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "data" / "parrotts_cache"),
        help="Directory containing .wav files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing .visemes.json files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel Rhubarb workers",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"ERROR: cache dir does not exist: {cache_dir}")
        return 1

    wav_files = sorted(cache_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {cache_dir}")
        return 0

    total = len(wav_files)
    computed = 0
    skipped = 0
    failed = 0

    from concurrent.futures import ProcessPoolExecutor

    def process_one(wav_path: Path) -> tuple[Path, bool, str]:
        viseme_path = wav_path.with_suffix(".visemes.json")
        if viseme_path.exists() and not args.force:
            return (wav_path, True, "cached")
        try:
            cues = _run_rhubarb(wav_path)
            viseme_path.write_text(json.dumps(cues, indent=2))
            return (wav_path, True, "computed")
        except Exception as e:
            return (wav_path, False, str(e))

    print(f"Processing {total} WAV files with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (wav_path, ok, msg) in enumerate(pool.map(process_one, wav_files), 1):
            if not ok:
                failed += 1
                print(f"  [{i}/{total}] FAIL {wav_path.name}: {msg}")
            elif msg == "cached":
                skipped += 1
                print(f"  [{i}/{total}] SKIP {wav_path.name}")
            else:
                computed += 1
                print(f"  [{i}/{total}] OK   {wav_path.name}")

    print()
    print(f"Done: computed={computed} skipped={skipped} failed={failed} total={total}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
