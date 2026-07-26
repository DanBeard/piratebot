# PirateBot ESP32 Prop Firmware

PlatformIO C/C++ firmware for ESP32 prop controllers.

## Supported boards

- `esp32-s3-devkitc-1`
- `esp32-c6-devkitc-1`

## Install PlatformIO

```bash
pip install platformio
```

## Build

```bash
cd props/firmware/esp32/platformio
pio run -e esp32-s3
```

## Flash

```bash
pio run -e esp32-s3 --target upload --upload-port /dev/ttyUSB0
```

## Configuration

Compile-time defaults are in `src/config.h`. Override at build time:

```bash
pio run -e esp32-s3 --build-flag "-DMESH_NODE_ID=\"cannon_01\"" \
    --build-flag "-DWIFI_SSID=\"YourSSID\"" \
    --build-flag "-DWIFI_PASS=\"YourPassword\"" \
    --build-flag "-DBROKER_HOST=\"192.168.0.50\""
```

Or store credentials in `data/config.json` and upload the filesystem:

```json
{
  "wifi_ssid": "YourSSID",
  "wifi_pass": "YourPassword",
  "broker_host": "192.168.0.50",
  "broker_ws_port": 9001,
  "profile": "cannon"
}
```

### NVS profile selection

The firmware reads the `profile` key from NVS (`piratebot` namespace).
You can set it with a simple tool or by re-flashing with a profile
hard-coded. Built-in profiles:

- `cannon` — GPIO 12, 0.5s pulse, 5s cooldown
- `smoke` — GPIO 13, 2s burst, 3s cooldown
- `strobe` — GPIO 14, 50/50ms blink x10, 2s cooldown
- `thunder` — GPIO 15, 0.2s flash, 1.5s cooldown
- `relay` — GPIO 16, 1s toggle
- `pir` — GPIO 25 PIR sensor
- `beam` — GPIO 26 beam-break sensor

Multiple profiles can be combined by setting `profile` to a
comma-separated list in NVS, e.g. `cannon,smoke,pir`.

## Protocol

The firmware speaks the PirateBot prop mesh protocol:

- WebSocket client to broker (port 9001)
- Optional MQTT fallback
- Sends `prop.state.announce` on connect and `prop.state.heartbeat` every 5s
- Subscribes to `scene.estop` and configured `effects.*` topics
- Uses framed binary envelope with JSON payload (CBOR reserved)

## Safety

- `max_on_ms` hard-off per profile
- `cooldown_ms` between firings
- `expire_ms` honored on incoming messages
- `scene.estop` stops all outputs immediately
- Watchdog enabled by default via Arduino core

## Adding CBOR

The codec currently uses JSON for framed messages. To enable CBOR, add a
library such as `bblanchon/ArduinoJson` with MessagePack support (already
included) or a dedicated CBOR library, then implement `Message::toCbor`
and `Message::fromCbor` in `src/codec/message.cpp`.

## Adding ESP-NOW

The transport layer is split so ESP-NOW can be added later:

1. Create `src/transports/espnow_client.h/.cpp`
2. Register peers in `setup()`
3. Flood-mesh relay uses the `FLAG_FLOOD_MESH` bit and a TTL counter
   (reserved; implement in `Message` framing once peers are known)
