# PirateBot Prop Mesh

PirateBot can join your distributed prop network so the portrait, fog
machines, strobes, thunder speakers, and any ESP32 / Raspberry Pi props
can trigger each other. This keeps the show reactive and lets you
rearrange sensors and effects every year without rewiring.

## Quick start

Enable the mesh in `config.portrait.yaml`:

```yaml
prop_mesh:
  enabled: true
  source: "piratebot"
  mode: "server"          # server | client | both
  host: "0.0.0.0"
  port: 9001
  # broker_url: "ws://192.168.0.100:9001/ws"  # for client/both mode
  auto_triggers:
    smoke.burst:
      - emotion: menacing
    thunder.clap:
      - emotion: surprised
      - emotion: angry
    strobe.flash:
      - emotion: surprised
    fog.thick:
      - tag: prop:fog
```

In `server` mode PirateBot hosts the mesh at `ws://<porch-pc-ip>:9001/ws`.
Any prop can connect and send/receive JSON events.

## Protocol

Every message is a JSON object:

```json
{
  "type": "thunder.clap",
  "source": "piratebot",
  "target": null,
  "payload": {},
  "timestamp": 1785079673.44
}
```

| Field | Meaning |
|-------|---------|
| `type` | Event name. Namespaced with dots, e.g. `smoke.burst`. |
| `source` | Which prop emitted it. |
| `target` | Optional prop ID. `null` means broadcast to all. |
| `payload` | Free-form data for the event. |
| `timestamp` | Unix time. |

## Standard events emitted by PirateBot

| Event | When |
|-------|------|
| `pirate.arrival` | A person arrives at the porch. |
| `pirate.departure` | A tracked person leaves. |
| `pirate.speak` | The pirate starts speaking a line. Payload includes `line_id`, `text`, `emotion`, `tags`. |
| `pirate.idle_speak` | Idle line played. |
| `smoke.burst` | Trigger fog/smoke machine. |
| `fog.thick` | Trigger heavy fog / long fog burst. |
| `thunder.clap` | Trigger thunder sound + light. |
| `strobe.flash` | Trigger strobe light. |

## Events PirateBot listens to

| Event | Action |
|-------|--------|
| `pirate.speak` | Play a specific audio URL + visemes if provided. |
| `pirate.expression` | Set expression, e.g. `surprised`. |
| `pirate.animation` | Play animation, e.g. `nod`, `shake_head`. |
| `pirate.gaze` | Look at `x`, `y` in screen coordinates. |

This lets a motion-sensor prop make the portrait react without going
through the full detection pipeline.

## Trigger sources

A prop event can be fired three ways:

1. **Voice line tag**: add `prop:<event>` to a line's `tags` in
   `data/voice_lines.yaml`.
2. **Emotion auto-trigger**: configure `prop_mesh.auto_triggers` in
   YAML; any line with that emotion fires the mapped events.
3. **Incoming mesh event**: another prop sends an event and PirateBot's
   local handler reacts.

## ESP32 / MicroPython skeleton

Connect to the mesh from an ESP32 and fire a smoke burst when a PIR
sensor triggers:

```python
import network
import uasyncio as asyncio
import uwebsockets.client as ws

WIFI_SSID = "your-ssid"
WIFI_PASS = "your-pass"
MESH_URL = "ws://192.168.0.50:9001/ws"  # PirateBot porch PC IP
PIR_PIN = 4

async def main():
    # Connect WiFi
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASS)
    while not wlan.isconnected():
        await asyncio.sleep(1)

    # Connect to mesh
    websocket = ws.connect(MESH_URL)

    from machine import Pin
    pir = Pin(PIR_PIN, Pin.IN)

    while True:
        if pir.value():
            msg = '{"type":"smoke.burst","source":"pir_prop_01","target":null,"payload":{"reason":"motion"},"timestamp":0}'
            websocket.send(msg)
            await asyncio.sleep(5)  # cooldown
        await asyncio.sleep(0.1)

asyncio.run(main())
```

For a Raspberry Pi Zero use any Python WebSocket client (`websockets` or
`aiohttp`).

## Wiring ideas

- **Thunder speaker prop**: listens for `thunder.clap`, plays a WAV and
  flashes an LED strip.
- **Fog machine prop**: listens for `smoke.burst` / `fog.thick`, opens a
  relay for 2–5 seconds.
- **Strobe prop**: listens for `strobe.flash`, flashes a relay-driven
  strobe.
- **Pressure mat prop**: emits `pirate.gaze` so the portrait looks toward
  the door.
- **Webcam prop**: emits `pirate.arrival` from a different angle, letting
  the portrait speak even if the porch PC's camera doesn't see the kid
  yet.

## Docker / network notes

If PirateBot runs in a container or behind a firewall, expose port 9001
(or whatever `prop_mesh.port` is). The mesh is plain WebSocket; add TLS
only if you expose it beyond your home network.

## Future hooks

- Add MQTT transport alongside WebSocket for very low-power props.
- Add `prop.state` heartbeat so the dashboard can show which props are
  online.
- Add `sequencer` events like `scene.haunted_painting` that trigger
  multiple props with delays.
