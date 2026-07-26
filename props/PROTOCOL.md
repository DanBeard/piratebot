# Prop Mesh Protocol

## Message envelope

Every message on the mesh is a JSON object with this envelope:

```json
{
  "topic": "effects.thunder.clap",
  "source": "piratebot",
  "target": null,
  "payload": {
    "duration_ms": 500,
    "intensity": 0.8
  },
  "timing": {
    "delay_ms": 150,
    "at_ts": null,
    "expire_ms": null
  },
  "meta": {
    "seq": 42,
    "session": "halloween-2026",
    "codecs": ["json", "cbor"]
  },
  "timestamp": 1785079673.44
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `topic` | string | Hierarchical topic, dot-separated. |
| `source` | string | ID of the prop/node that emitted the message. |
| `target` | string \| null | Optional destination prop ID; `null` is broadcast. |
| `payload` | object | Event-specific data. |
| `timing.delay_ms` | int \| null | Wait this many ms before acting. Used to sync across props. |
| `timing.at_ts` | float \| null | Absolute Unix timestamp to act at. Overrides `delay_ms` if set. |
| `timing.expire_ms` | int \| null | Drop the event if it cannot be executed within this window. |
| `meta.seq` | int | Monotonic sequence number from the source. |
| `meta.session` | string | Show/session ID, useful for avoiding stale events. |
| `meta.codecs` | list[string] | Codecs the sender supports: `["json", "cbor"]` by default. |
| `timestamp` | float | When the message was emitted. |

## Topic conventions

Topics are dot-separated and read from left to right:

- `pirate.*` — events produced or consumed by the PirateBot portrait.
- `effects.*` — requests to physical effects (fog, thunder, strobe, etc.).
- `sensors.*` — events from sensors (PIR, pressure mat, beam break, etc.).
- `scene.*` — high-level scene or sequence commands.
- `prop.state.*` — heartbeats and capability announcements.

## Standard topics

### PirateBot portrait

| Topic | Direction | Payload |
|-------|-----------|---------|
| `pirate.arrival` | emit | `{track_id: int}` |
| `pirate.departure` | emit | `{track_id: int}` |
| `pirate.speak` | emit/handle | `{line_id, text, emotion, tags}` |
| `pirate.idle_speak` | emit | `{line_id, text}` |
| `pirate.expression` | handle | `{expression: "surprised"}` |
| `pirate.animation` | handle | `{animation: "nod", loop: false}` |
| `pirate.gaze` | handle | `{x: 0.5, y: 0.5}` |

### Effects

| Topic | Payload |
|-------|---------|
| `effects.smoke.burst` | `{duration_ms: 2000, intensity: 0.5}` |
| `effects.fog.thick` | `{duration_ms: 5000}` |
| `effects.thunder.clap` | `{duration_ms: 800, flash_count: 3}` |
| `effects.strobe.flash` | `{duration_ms: 500, frequency_hz: 10}` |
| `effects.light.color` | `{color: "#ff6600", duration_ms: 1000}` |
| `effects.servo.move` | `{id: "chest", angle: 90, speed_ms: 200}` |
| `effects.audio.play` | `{file: "scream.wav", volume: 0.8}` |

### Sensors

| Topic | Payload |
|-------|---------|
| `sensors.pir.motion` | `{zone: "porch_left", confidence: 1.0}` |
| `sensors.pressure.mat` | `{zone: "step", pressed: true}` |
| `sensors.beam.break` | `{zone: "walkway"}` |

### Scenes

| Topic | Payload |
|-------|---------|
| `scene.start` | `{scene: "haunted_painting", intensity: 0.7}` |
| `scene.stop` | `{scene: "haunted_painting"}` |
| `scene.pause` | `{}` |

### Prop state

| Topic | Payload |
|-------|---------|
| `prop.state.announce` | `{id, capabilities: ["effects.thunder.clap", "sensors.pir.motion"], codecs: ["json", "cbor"]}` |
| `prop.state.heartbeat` | `{id, uptime_s, load}` |
| `prop.state.error` | `{id, error, topic}` |
| `prop.state.topics` | `{map: {"effects.thunder.clap": 1, ...}}` (binary topic-ID assignment) |

## Timing rules

1. If `timing.at_ts` is set, schedule the action for that exact timestamp.
2. Else if `timing.delay_ms` is set, wait that long after receiving.
3. Else execute immediately.
4. If `timing.expire_ms` is set and the scheduled time is already in the
   past by more than that amount, drop the event.

This lets the publisher schedule synchronized multi-prop effects without
needing every prop's clock to be perfect.

## Serialization

### JSON
Default transport. Human-readable, works everywhere.

### CBOR
Compact binary representation of the same envelope. Props and clients
negotiate support by advertising `meta.codecs` in `prop.state.announce`.
CBOR is recommended for bandwidth-constrained transports such as ESP-NOW
or crowded 2.4 GHz WiFi.

### Framed binary
For very constrained links, messages can be sent in a tiny framed
envelope:

```
[magic:2][version:1][flags:1][codec:1][payload_len:2][payload:N]
```

- `magic` is always `0x50 0x42` ("PB").
- `version` is `0x01`. New transports must bump this only when the header
  shape changes.
- `flags` bits:
  - `0x01` `FLAG_FLOOD_MESH` — message may be relayed up to N hops
    (future ESP-NOW flood mesh).
  - `0x02` `FLAG_TLV_BINARY` — payload is a custom packed format instead
    of CBOR (future).
  - Remaining bits reserved.
- `codec` byte: `0x01` JSON, `0x02` CBOR, `0x03`+ reserved.
- `payload_len` is a 16-bit unsigned big-endian integer. Max payload is
  65535 bytes.

The framed format lets receivers detect the protocol without deep
inspection. The `version` and `flags` bytes make future flood-mesh and
binary-schema extensions easy to add without breaking older props.

### Prop configuration

| Topic | Direction | Payload |
|-------|-----------|---------|
| `prop.config.get` | handle | `{}` |
| `prop.config.current` | emit | `{node_id, profile, wifi_ssid, broker_host, broker_ws_port, broker_mqtt_port, ota_enabled}` |
| `prop.config.set` | handle | `{profile, wifi_ssid, wifi_pass, broker_host, broker_ws_port, broker_mqtt_port, ota_enabled}` |
| `prop.config.ack` | emit | `{success, needs_reboot}` |

### Prop OTA

| Topic | Direction | Payload |
|-------|-----------|---------|
| `prop.ota.enable` | handle | `{enabled: bool, confirm: true}` |
| `prop.ota.start` | handle | `{url: "http://.../firmware.bin"}` |
| `prop.ota.status` | emit | `{enabled, error, success, bytes_written}` |

`prop.ota.enable` requires `confirm: true` and only changes state; it does not
flash. `prop.ota.start` only runs if OTA is enabled and the current scene is
`idle`, `quiet`, or `estop`. This prevents accidental firmware updates during
a live show.

## Discovery

The broker answers UDP broadcast discovery requests on a configurable
port. A prop can send:

```json
{"cmd": "discover", "session": "halloween-2026"}
```

and receive:

```json
{"broker": "ws://192.168.0.50:9001/ws", "session": "halloween-2026"}
```

## Transport notes

- **WebSocket**: reliable, ordered, star topology. Good for the portrait
  and Raspberry Pi props. The default codec is JSON; CBOR can be
  negotiated via `prop.state.announce`.
- **MQTT**: add a broker bridge for existing ESP32 props, smart plugs,
  and Zigbee2MQTT devices that already speak MQTT.
- **UDP multicast**: broadcast discovery and fire-and-forget events where
  ordering doesn't matter.
- **ESP-NOW**: low-latency peer-to-peer between ESP32 props. Use the
  framed binary format, and reserve the `FLAG_FLOOD_MESH` flag for
  controlled multi-hop relaying (max TTL to be defined by the transport).

All transports carry the same logical envelope. Transports that cannot
represent the full envelope must map losslessly to the canonical JSON
form at the nearest bridge.
