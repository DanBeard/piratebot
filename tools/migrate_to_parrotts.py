#!/usr/bin/env python3
"""One-shot migration: seed the cluster parrotts service from piratebot data.

Steps:

1. POST the pirate character using ``data/pirate_voice_sample.{wav,txt}``.
2. Walk ``data/voice_lines.yaml`` and submit a single batch of all lines to
   ``/v1/generate/batch``.
3. Long-poll the parent job until it terminates, printing per-status counts.
4. Report any children that failed.

Re-running is safe and cheap — parrotts content-addresses each line by
``(character, text, engine, settings)``, so cache hits resolve immediately
without re-running the engine.

Usage:

    python tools/migrate_to_parrotts.py [--base-url URL] [--character NAME]
                                        [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from services.parrotts_vendor import ParrottsClient, ParrottsError, ParrottsTimeout  # noqa: E402


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
                if entry.get("tags"):
                    line["tags"] = list(entry["tags"])
                if entry.get("emotion"):
                    line["emotion"] = entry["emotion"]
                out.append(line)
    return out


def ensure_character(
    client: ParrottsClient,
    name: str,
    display_name: str,
    reference_audio: Path,
    reference_transcript: Path,
) -> None:
    existing = client.get_character(name)
    if existing is not None:
        print(f"  character {name!r} already registered ({existing.get('display_name')})")
        return
    print(f"  registering character {name!r} from {reference_audio.name}")
    transcript = reference_transcript.read_text().strip()

    class _Spec:
        pass

    spec = _Spec()
    spec.name = name
    spec.display_name = display_name
    spec.default_engine = "sovits"
    spec.voice_settings = {}
    spec.tags = ["pirate"]

    client.register_character(
        spec=spec,
        reference_audio=reference_audio,
        reference_transcript=transcript,
        overwrite=False,
    )
    print(f"  registered {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:18003",
                        help="parrotts base URL (default: http://localhost:18003 — assumes "
                             "a port-forward is active)")
    parser.add_argument("--character", default="pirate")
    parser.add_argument("--display-name", default="Captain Barnacle Bill")
    parser.add_argument("--reference-audio",
                        default=str(REPO_ROOT / "data/pirate_voice_sample.wav"))
    parser.add_argument("--reference-transcript",
                        default=str(REPO_ROOT / "data/pirate_voice_sample.txt"))
    parser.add_argument("--yaml", default=str(REPO_ROOT / "data/voice_lines.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be sent without contacting parrotts")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Seconds to wait for the batch to finish (default: 1 hour)")
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    lines = collect_lines_from_yaml(yaml_path)
    print(f"Loaded {len(lines)} lines from {yaml_path}")

    if args.dry_run:
        from collections import Counter
        c = Counter((l["category"], l.get("subcategory") or "") for l in lines)
        for (cat, sub), n in sorted(c.items()):
            label = f"{cat}/{sub}" if sub else cat
            print(f"  {label}: {n}")
        return 0

    print(f"Connecting to parrotts at {args.base_url}")
    client = ParrottsClient(base_url=args.base_url, timeout=120.0)
    try:
        info = client._http.get("/v1/info").json()
        print(f"  parrotts {info.get('version')} — engines {info.get('engines_loaded')} "
              f"— {info.get('line_count')} lines already in library")
    except Exception as e:
        print(f"ERROR: cannot reach parrotts at {args.base_url}: {e}")
        return 1

    print()
    print("==> ensure_character")
    try:
        ensure_character(
            client,
            name=args.character,
            display_name=args.display_name,
            reference_audio=Path(args.reference_audio),
            reference_transcript=Path(args.reference_transcript),
        )
    except ParrottsError as e:
        print(f"ERROR: register_character failed: {e}")
        return 1

    print()
    print(f"==> submit batch ({len(lines)} lines)")
    t0 = time.monotonic()
    try:
        parent = client.generate_batch(
            character=args.character,
            lines=lines,
            timeout=args.timeout,
            include_children=True,
        )
    except ParrottsTimeout as e:
        print(f"TIMEOUT after {time.monotonic() - t0:.0f}s: {e}")
        print("Re-run the script — content-addressing makes cache hits free.")
        return 2
    except ParrottsError as e:
        print(f"ERROR: batch submit failed: {e}")
        return 1

    elapsed = time.monotonic() - t0
    summary = parent.get("summary", {})
    print(f"  parent {parent['job_id']} → status={parent['status']} in {elapsed:.0f}s")
    print(f"  done={summary.get('done', 0)} failed={summary.get('failed', 0)} "
          f"queued={summary.get('queued', 0)} running={summary.get('running', 0)} "
          f"total={summary.get('total', 0)}")

    failed = summary.get("failed", 0)
    if failed:
        print()
        print(f"==> {failed} children failed; inspecting:")
        for child_id in parent.get("children", []):
            r = client._http.get(f"/v1/jobs/{child_id}")
            if r.status_code != 200:
                continue
            job = r.json()
            if job.get("status") == "failed":
                print(f"  {child_id} text={job.get('text', '')[:60]!r} "
                      f"error={job.get('error', '')[:120]}")

    return 0 if failed == 0 and parent.get("status") == "done" else 3


if __name__ == "__main__":
    sys.exit(main())
