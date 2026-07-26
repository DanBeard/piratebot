# Helm (2021) — Legacy Prop Mesh Summary

**Date unzipped:** 2026-07-26  
**Original zip:** `Helm_10_31_21.zip`  
**Status:** Deprecated. Kept for inspiration only.

## What this was

A 2021 Halloween pirate-ship show called **Helm**. It coordinated
multiple Raspberry Pi props over a custom WebSocket mesh with UDP
multicast discovery. A central broker (`MainDeck`) routed messages
between "crewmates" (prop controllers) and a browser UI let the operator
fire effects manually.

## Key components

### `crewmates/main_deck.py`
The WebSocket broker. Ran on `ws://0.0.0.0:31337`.
- Maintained a `manifest` list of connected crewmates.
- Routed messages by `address` with simple pubsub.
- Broadcasted the manifest to anyone subscribed to `"*"`.
- No persistence, no authentication, no TLS.

### `crewmates/util.py`
Shared mesh utilities.
- `MessageTypes`: `COMMAND`, `NOTIFY`, `SUBSCRIBE`, `LOG`, `GET_MANIFEST`, `MANIFEST`, `SET_ADDRESS`.
- UDP multicast discovery on `224.1.33.7:31338`.
- Clients shouted `AHOY`; servers replied `WELCOME ABOARD`.
- Shouting client tried ports 31338–31437 to avoid collisions.

### `crewmates/crewmate.py`
Base class for every prop node.
- `CrewmateProperty` descriptor: setting a property auto-broadcast a
  `NOTIFY` message.
- Each crewmate had an `address` (e.g. `CANNON`, `AMBIANCE`).
- Auto-discovery of broker via `shout_client()`.
- Commands, subscriptions, and property-change notifications.

## Prop controllers (`crewmates/`)

| Crewmate | Address | Hardware | Role |
|----------|---------|----------|------|
| `Cannon` | `CANNON` | GPIO + audio | Played cannon sound, flashed GPIO light for 0.5s, 3.5s cooldown, 2.5s lockout. |
| `Ambiance` | `AMBIANCE` | GPIO + `alsaaudio` + VLC | Looped background music; periodically played voice lines; dimmed volume when pumpkins sang. |
| `Pumpkins` | `PUMPKINS` | GPIO + VLC HTTP API | Played singing-pumpkin videos from `/home/pi/videos/` via VLC; cycled songs every 10 min; returned to idle loop. |
| `QuarterMaster` | `QUARTERMASTER` | — | Intended coordinator. Mostly set `state` property. Singing logic was commented out. |

## Operator UIs (`ui/`)

- `captain.html`: desktop dashboard with Fabric.js sprites for the
  house, ship, cannon, and quartermaster. Subscribed to all events.
- `mobile.html`: big-button mobile UI for firing the cannon and making
  pumpkins sing.
- `ambiance.html`: present but not inspected in detail.

Both UIs used WebSocket directly and updated button states based on the
manifest.

## Message format

```json
{
  "type": "C",
  "address": "CANNON",
  "from": "CAPTAIN_UI",
  "data": {"command": "FIRE"}
}
```

Notification example:

```json
{
  "type": "N",
  "address": "CANNON",
  "from": "CANNON",
  "data": {"prop": "firing", "val": true}
}
```

## Notable design ideas worth carrying forward

1. **Address-based pubsub.** Each prop has a human-readable ID; send to a
   specific prop or broadcast via `"*"`. Keep this in the new mesh.
2. **Manifest / heartbeat.** The broker knew who was online and beaconed
   it. The new system should keep this via `prop.state.announce` and
   `prop.state.heartbeat`.
3. **Property-change auto-notify.** Setting `self.firing = True` on the
   cannon automatically fired a `NOTIFY`. Useful pattern for stateful
   actuators.
4. **Volume ducking.** Ambiance quieted itself when Pumpkins sang. Good
   example of cross-prop courtesy behavior.
5. **UDP discovery.** Shouting + multicast lets props find the broker
   without hard-coded IPs. Keep, but make it optional and add fallback
   static config.
6. **State machine coordinator.** QuarterMaster concept of a central
   state machine is worth revisiting for `scene.*` topics in the new
   protocol.

## Notable limitations to avoid repeating

1. **No message envelope.** `type`/`address`/`from`/`data` was custom and
   non-extensible. The new protocol uses `topic`/`source`/`target`/
   `payload`/`timing`/`meta`.
2. **No timing fields.** Effects could not be scheduled across props.
   New protocol adds `delay_ms`/`at_ts` for synchronization.
3. **Broker IP via discovery only.** If UDP failed, nothing worked. New
   system should allow static broker URL fallback.
4. **One transport only.** WebSocket only. New system should support MQTT
   bridge for low-end ESP32s.
5. **Hard-coded command strings.** `"FIRE"`, `"SING"`, `"VOLUME"`. New
   system should use topic namespaces instead.
6. **No safety limits on actuation.** Cannon had a basic firing flag but
   no max-on-time, watchdog, or estop. New actuators should have
   `expire_ms` and hardware watchdogs.
7. **UI had to know every prop.** New system should allow generic UI
   elements driven by `prop.state.announce` capabilities.
8. **VLC dependency for video.** Worked but fragile. New Pumpkin-like prop
   should consider MPV or a dedicated video player with HTTP API.
9. **GPIO used BOARD mode.** New RPi code should prefer BCM or use
   `gpiozero` for portability.

## File inventory (non-venv)

```
Helm/
├── crewmates/
│   ├── ambiance.py
│   ├── cannon.py
│   ├── crewmate.py
│   ├── main_deck.py
│   ├── pumpkins.py
│   ├── quartermaster.py
│   └── util.py
├── ui/
│   ├── ambiance.html
│   ├── captain.html
│   ├── mobile.html
│   ├── server.sh
│   ├── js/
│   │   ├── cannon.js
│   │   ├── fabric.min.js
│   │   ├── pirateShip.js
│   │   └── quartermaster.js
│   ├── images/
│   │   ├── cannon.png
│   │   ├── fire_ball.png
│   │   ├── house.png
│   │   ├── plain_cannonball.png
│   │   ├── QUARTERMASTER.png
│   │   └── ship_sprite.png
│   └── sounds/
│       ├── nri-cannon.mp3
│       └── splash.mp3
├── crewmates/sounds/
│   ├── ambiance/music/
│   ├── ambiance/voice_lines/
│   ├── nri-cannon.mp3
│   └── splash.mp3
├── requirements.txt
├── reqs.sh
└── venv/  (excluded from this summary)
```

## Recommended extraction list for the new system

- Keep **address-based routing**, but call it `topic`.
- Keep **manifest/heartbeat**, map to `prop.state.*`.
- Keep **auto-notify properties** as an optional client helper.
- Keep **UDP discovery**, add static fallback.
- Keep **coordinator state machine**, implement as `scene.*` topic
  handler.
- Keep **audio ducking / cross-prop courtesy**, implement as policy
  handlers.
