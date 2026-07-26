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
    "session": "halloween-2026"
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
| `prop.state.announce` | `{id, capabilities: ["effects.thunder.clap", "sensors.pir.motion"]}` |
| `prop.state.heartbeat` | `{id, uptime_s, load}` |
| `prop.state.error` | `{id, error, topic}` |

## Timing rules

1. If `timing.at_ts` is set, schedule the action for that exact timestamp.
2. Else if `timing.delay_ms` is set, wait that long after receiving.
3. Else execute immediately.
4. If `timing.expire_ms` is set and the scheduled time is already in the
   past by more than that amount, drop the event.

This lets the publisher schedule synchronized multi-prop effects without
needing every prop's clock to be perfect.

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
  and Raspberry Pi props.
- **MQTT**: add a broker bridge for existing ESP32 props that already
  speak MQTT.
- **UDP multicast**: broadcast discovery and fire-and-forget events where
  ordering doesn't matter.

New transports must carry the same JSON envelope; no binary custom
protocol is planned for now.
