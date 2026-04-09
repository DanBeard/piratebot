"""Vendored parrotts HTTP client.

These files are copied verbatim from ``~/parrotts/parrotts/client/`` and
``~/parrotts/parrotts/_status.py``. The only local modification is changing
``from parrotts._status import ...`` to a relative import in ``client.py``
so the vendor package is self-contained.

To resync:

    cp ~/parrotts/parrotts/_status.py services/parrotts_vendor/_status.py
    cp ~/parrotts/parrotts/client/parrotts_client.py services/parrotts_vendor/client.py
    sed -i 's|from parrotts._status|from ._status|' services/parrotts_vendor/client.py

This vendoring exists so piratebot doesn't need to depend on the full parrotts
pip package (which pulls torch, chatterbox-tts, and the FastAPI server).
"""

from .client import (
    GenerateResult,
    ParrottsClient,
    ParrottsError,
    ParrottsTimeout,
)

__all__ = [
    "GenerateResult",
    "ParrottsClient",
    "ParrottsError",
    "ParrottsTimeout",
]
