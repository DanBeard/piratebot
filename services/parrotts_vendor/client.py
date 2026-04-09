"""Thin sync HTTP client for parrotts.

Hides the enqueue + poll loop behind ``generate(character, text)`` so callers
just see a blocking call that returns a line id and audio URL (or raises on
error / timeout).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from ._status import DONE, FAILED


class CharacterSpecLike(Protocol):
    name: str
    display_name: str


class ParrottsError(Exception):
    """Server returned an error or job failed."""


class ParrottsTimeout(ParrottsError):
    """Job did not finish within the requested timeout."""


@dataclass
class GenerateResult:
    job_id: str
    line_id: str
    audio_url: str
    cached: bool


def _raise_for(response: httpx.Response, label: str) -> None:
    if response.status_code >= 400:
        raise ParrottsError(f"{label}: {response.status_code} {response.text}")


class ParrottsClient:
    def __init__(self, base_url: str = "http://parrotts:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ----- characters -----

    def list_characters(self) -> list[dict[str, Any]]:
        r = self._http.get("/v1/characters")
        r.raise_for_status()
        return r.json()["characters"]

    def get_character(self, name: str) -> dict[str, Any] | None:
        r = self._http.get(f"/v1/characters/{name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def register_character(
        self,
        spec: "CharacterSpecLike",
        reference_audio: Path,
        reference_transcript: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Register a character.

        ``spec`` may be a ``parrotts.specs.CharacterSpec`` (when running inside
        the parrotts package) or any object with the attributes ``name``,
        ``display_name``, ``default_engine``, ``voice_settings``, ``tags`` —
        so consumers can vendor the client without importing pydantic.
        """
        with reference_audio.open("rb") as f:
            files = {"reference_audio": (reference_audio.name, f, "audio/wav")}
            data = {
                "name": spec.name,
                "display_name": spec.display_name,
                "reference_transcript": reference_transcript,
                "default_engine": getattr(spec, "default_engine", "chatterbox"),
                "voice_settings_json": json.dumps(getattr(spec, "voice_settings", {}) or {}),
                "tags_csv": ",".join(getattr(spec, "tags", []) or []),
                "overwrite": "true" if overwrite else "false",
            }
            r = self._http.post("/v1/characters", data=data, files=files)
        _raise_for(r, "register_character")
        return r.json()

    def delete_character(self, name: str) -> None:
        r = self._http.delete(f"/v1/characters/{name}")
        if r.status_code == 404:
            return
        _raise_for(r, "delete_character")

    # ----- generate -----

    def generate(
        self,
        character: str,
        text: str,
        engine: str | None = None,
        voice_settings: dict[str, Any] | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        tags: list[str] | None = None,
        emotion: str | None = None,
        force: bool = False,
        timeout: float = 300.0,
    ) -> GenerateResult:
        """Enqueue a generation job and block (via long-poll) until it
        finishes or ``timeout`` seconds elapse. Raises ``ParrottsError`` on
        failure or ``ParrottsTimeout`` on deadline."""
        body: dict[str, Any] = {"character": character, "text": text}
        if engine is not None:
            body["engine"] = engine
        if voice_settings is not None:
            body["voice_settings"] = voice_settings
        if category is not None:
            body["category"] = category
        if subcategory is not None:
            body["subcategory"] = subcategory
        if tags is not None:
            body["tags"] = tags
        if emotion is not None:
            body["emotion"] = emotion
        if force:
            body["force"] = True

        r = self._http.post("/v1/generate", json=body)
        _raise_for(r, "generate")
        job = r.json()
        return self._poll_until_done(job["job_id"], timeout=timeout)

    def generate_batch(
        self,
        character: str,
        lines: list[dict[str, Any]],
        engine: str | None = None,
        voice_settings: dict[str, Any] | None = None,
        force: bool = False,
        timeout: float = 1800.0,
        include_children: bool = False,
    ) -> dict[str, Any]:
        """Submit a batch and block until the parent reaches a terminal state.

        Returns the parent job dict (with ``summary`` aggregating child counts).
        Pass ``include_children=True`` to also fetch the full child id list
        on the final poll.
        """
        body: dict[str, Any] = {"character": character, "lines": lines}
        if engine is not None:
            body["engine"] = engine
        if voice_settings is not None:
            body["voice_settings"] = voice_settings
        if force:
            body["force"] = True

        r = self._http.post("/v1/generate/batch", json=body)
        _raise_for(r, "generate_batch")
        parent = r.json()
        return self._poll_parent_until_done(
            parent["job_id"],
            timeout=timeout,
            include_children=include_children,
        )

    def _poll_parent_until_done(
        self, parent_id: str, timeout: float, include_children: bool
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ParrottsTimeout(f"batch parent {parent_id} did not finish within timeout")
            wait = min(remaining, 60.0)
            try:
                r = self._http.get(
                    f"/v1/jobs/{parent_id}",
                    params={"wait": f"{wait:.2f}", "include_children": include_children},
                    timeout=wait + 5.0,
                )
            except httpx.TransportError:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            if r.status_code == 404:
                raise ParrottsError(f"batch parent {parent_id} not found")
            r.raise_for_status()
            parent = r.json()
            if parent.get("status") in (DONE, FAILED):
                return parent
            # Loop and long-poll again.

    def _poll_until_done(self, job_id: str, timeout: float) -> GenerateResult:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ParrottsTimeout(f"job {job_id} did not finish within timeout")
            wait = min(remaining, 60.0)
            try:
                r = self._http.get(
                    f"/v1/jobs/{job_id}",
                    params={"wait": f"{wait:.2f}"},
                    timeout=wait + 5.0,
                )
            except httpx.TransportError:
                # Transient — server restart, brief network blip. Back off
                # briefly and try again until the deadline.
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            if r.status_code == 404:
                raise ParrottsError(f"job {job_id} not found")
            r.raise_for_status()
            job = r.json()
            status = job.get("status")
            if status == DONE:
                return GenerateResult(
                    job_id=job["job_id"],
                    line_id=job["line_id"],
                    audio_url=f"{self.base_url}/v1/lines/{job['line_id']}.wav",
                    cached=bool(job.get("cached")),
                )
            if status == FAILED:
                raise ParrottsError(f"job {job_id} failed: {job.get('error')}")
            # status is queued or running — long-poll again

    # ----- lines -----

    def search_lines(
        self,
        q: str,
        character: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
        top_k: int = 5,
        exclude_recent: float = 300.0,
    ) -> dict[str, Any]:
        """Semantic search over the line library. Returns the response
        envelope ``{"results": [{"line", "similarity"}, ...], "summary": {...}}``.

        ``summary.degraded == True`` means the embed service was unreachable
        and the results are a filter-only random sample (similarity=0).
        """
        params: dict[str, Any] = {
            "q": q,
            "top_k": top_k,
            "exclude_recent": exclude_recent,
        }
        if character is not None:
            params["character"] = character
        if category is not None:
            params["category"] = category
        if subcategory is not None:
            params["subcategory"] = subcategory
        r = self._http.get("/v1/lines/search", params=params)
        _raise_for(r, "search_lines")
        return r.json()

    def get_line(self, line_id: str) -> dict[str, Any] | None:
        """Metadata for a single line. Returns ``None`` on 404."""
        r = self._http.get(f"/v1/lines/{line_id}")
        if r.status_code == 404:
            return None
        _raise_for(r, "get_line")
        return r.json()

    def random_line(
        self,
        character: str | None = None,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> dict[str, Any] | None:
        """Pick a random line matching the filter. Returns ``None`` on 404."""
        params: dict[str, Any] = {}
        if character is not None:
            params["character"] = character
        if category is not None:
            params["category"] = category
        if subcategory is not None:
            params["subcategory"] = subcategory
        r = self._http.get("/v1/lines/random", params=params)
        if r.status_code == 404:
            return None
        _raise_for(r, "random_line")
        return r.json()

    def get_visemes(self, line_id: str) -> list[dict[str, Any]] | None:
        """Fetch (and on first call generate) Rhubarb mouth cues for a line.

        Returns a list of ``{"shape", "start_time", "end_time"}`` dicts, or
        ``None`` if the line is unknown. Raises ``ParrottsError`` if Rhubarb
        is unavailable on the server.
        """
        r = self._http.get(f"/v1/lines/{line_id}/visemes")
        if r.status_code == 404:
            return None
        _raise_for(r, "get_visemes")
        return r.json()["visemes"]

    def download_line(self, line_id: str, dest: Path) -> Path:
        """Stream a generated audio file to ``dest``."""
        with self._http.stream("GET", f"/v1/lines/{line_id}.wav") as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    # ----- cleanup -----

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ParrottsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
