#!/usr/bin/env python3
"""
Raspberry Pi skeleton for a PirateBot prop node.

Run on a Pi Zero (or any Pi) connected to relays, LEDs, speakers, etc.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from props.lib.mesh_client import MeshClient, MeshMessage

logger = logging.getLogger("rpi_prop")

PROP_ID = "rpi_thunder_01"
BROKER_URL = "ws://192.168.0.50:9001/ws"


def on_thunder(message: MeshMessage) -> None:
    """Handle a thunder clap request."""
    duration_ms = message.payload.get("duration_ms", 800)
    logger.info(f"THUNDER for {duration_ms}ms")
    # Trigger relay / audio here


def on_strobe(message: MeshMessage) -> None:
    duration_ms = message.payload.get("duration_ms", 500)
    logger.info(f"STROBE for {duration_ms}ms")
    # Trigger strobe relay here


async def heartbeat(client: MeshClient) -> None:
    start = time.time()
    while True:
        await client.publish(
            "prop.state.heartbeat",
            {"uptime_s": int(time.time() - start), "load": 0.1},
        )
        await asyncio.sleep(30)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)

    client = MeshClient(broker_url=BROKER_URL, source=PROP_ID)
    client.subscribe("effects.thunder.clap", on_thunder)
    client.subscribe("effects.strobe.flash", on_strobe)

    await client.connect()

    await client.publish(
        "prop.state.announce",
        {"capabilities": ["effects.thunder.clap", "effects.strobe.flash"]},
    )

    await heartbeat(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
