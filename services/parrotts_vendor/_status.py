"""Job status constants — a leaf module with no other parrotts imports.

Lives in its own file so the client (``parrotts.client.parrotts_client``) can
import the constants without dragging in the server-side machinery (asyncio,
sqlite3, pydantic-settings) that ``parrotts.jobs`` and ``parrotts.config``
pull in transitively. The smoke-test debugging that exposed this lives in
PHASE3_DEFERRED.md item #7.
"""

from __future__ import annotations

from typing import Literal

JobStatus = Literal["queued", "running", "done", "failed"]

QUEUED: JobStatus = "queued"
RUNNING: JobStatus = "running"
DONE: JobStatus = "done"
FAILED: JobStatus = "failed"

TERMINAL_STATUSES: tuple[JobStatus, ...] = (DONE, FAILED)
