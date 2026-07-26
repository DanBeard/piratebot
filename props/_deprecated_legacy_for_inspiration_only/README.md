# ⚠️ Deprecated Legacy Prop Code — For Inspiration Only

This directory is a **drop site for old prop firmware / broker code** that
may still contain useful ideas, but it is **not** wired into the current
PirateBot prop mesh.

## Why this exists

The Halloween prop system has evolved over several years. Older ESP32,
Raspberry Pi, and custom pubsub code may have good tricks for:

- Reliable UDP broadcast patterns
- Debouncing specific sensors
- Relay timing and safety interlocks
- Low-latency local sequences
- Power supply and wiring lessons learned the hard way

Rather than deleting that history, we keep it here as a reference while
we rebuild a cleaner, more capable, unified system under the rest of
`props/`.

## Rules of engagement

1. **Do not import from here.** Nothing in this directory should be
   referenced by production code in `props/broker/`, `props/lib/`,
   `services/`, or `main.py`.
2. **Do not run this code directly** without reviewing it first. It
   may be out of date, use old topic names, or rely on hardware that no
   longer exists.
3. **Document anything you extract.** If a snippet from here inspires a
   new feature, add a note to `props/ARCHITECTURE.md` or
   `props/docs/wiring_guide.md` explaining the lineage.

## Target structure

Organize legacy files by year and platform so we can find things later:

```
_deprecated_legacy_for_inspiration_only/
├── 2023/
│   ├── esp32_fog_relay.ino
│   └── rpi_thunder_player.py
├── 2024/
│   ├── custom_pubsub_broker.py
│   └── esp32_pir_mesh.py
└── 2025/
    └── README.md
```

## New system

The rebuilt prop mesh lives at the top level of `props/`:

- `props/PROTOCOL.md` — current message format and topic conventions
- `props/broker/mesh_broker.py` — reference broker
- `props/lib/mesh_client.py` — async Python client
- `props/firmware/` — current firmware skeletons

Anything new should target that system, not this directory.
