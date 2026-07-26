# Prop Mesh Architecture

## Overview

```
                +------------------+
                |   PirateBot PC   |
                | (portrait + VLM +|
                |  YOLO + broker)  |
                +--------+---------+
                         |
      +------------------+------------------+
      |                  |                  |
  WebSocket          MQTT bridge        UDP multicast
      |                  |                  |
 +----+-----+      +------+------+    +-----+------+
 | portrait |      |  fog prop   |    |  discovery |
 | browser  |      |  (ESP32)    |    |  broadcast|
 +----------+      +-------------+    +------------+

       +------------+       +-------------+
       | thunder Pi |       | pressure mat|
       |   (Rpi)    |       |   (ESP32)   |
       +------------+       +-------------+
```

## Components

### 1. Broker (`props/broker/mesh_broker.py`)

The reference broker is a Python process that:
- Accepts WebSocket connections on `ws://host:port/ws`.
- Optionally bridges to an MQTT broker on `mqtt://host:port`.
- Optionally listens for UDP multicast discovery requests.
- Maintains a subscriber registry so it can route by topic.
- Broadcasts `prop.state.*` to a management channel.

It is **not** the only possible broker. A Raspberry Pi, a router with
MQTT, or even one of the props can run the broker.

### 2. PirateBot integration (`services/prop_controller.py`)

PirateBot does not host the broker by default. Instead it connects as a
client to `prop_mesh.broker_url` and:
- Publishes `pirate.*` topics on track lifecycle and speech.
- Subscribes to `pirate.expression`, `pirate.animation`, `pirate.gaze`,
  and any `scene.*` topics.
- Translates voice-line tags (`prop:effects.thunder.clap`) and emotions
  into mesh topics.

### 3. Client library (`props/lib/mesh_client.py`)

Python async client used by PirateBot and Raspberry Pi props. Features:
- Auto-reconnect.
- Sequence numbers.
- Local scheduler for `delay_ms` / `at_ts`.
- Subscription helpers: `client.subscribe("effects.thunder.*", handler)`.

### 4. Firmware skeletons

- `props/firmware/esp32/skeleton.py` — MicroPython.
- `props/firmware/rpi/skeleton.py` — regular Python.

Both include WiFi connection, broker discovery, heartbeat, and a simple
effect loop.

## Design principles

1. **No hard-coded effect mapping.** Props announce their capabilities;
   scenes and PirateBot publish topics. The same firmware can be a fog
   machine one year and a strobe the next by changing config.
2. **Time-aware scheduling.** Network jitter is normal; compensate with
   explicit delays and absolute timestamps.
3. **Backpressure-free for actuation.** Sensors can emit many events;
   actuators should debounce and drop stale events, never queue forever.
4. **Observable.** Every prop sends a heartbeat and announces its
   capabilities at startup.
