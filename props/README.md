# PirateBot Prop Mesh

A modern, capable pubsub network for coordinating Halloween props: the
portrait, fog machines, thunder speakers, strobes, pressure mats, PIRs,
servos, and anything else that can speak JSON.

This directory contains the shared protocol, reference broker, client
libraries, and skeleton firmware for ESP32 and Raspberry Pi props.

## Goals

- **Decoupled**: props announce what they can do; other props subscribe
to events by topic. Rearrange the show every year by changing config,
not wiring.
- **Multi-transport**: WebSocket star broker, MQTT bridge, UDP multicast
for local discovery and fallback broadcast, and local Unix sockets for
same-machine integration.
- **Time-aware**: events can carry `delay_ms` and `at_ts` so props can
schedule effects locally and compensate for WiFi jitter.
- **Observable**: heartbeats and `prop.state` messages make it easy to see
which props are alive from PirateBot or a dashboard.
- **Extensible**: new effects, sensors, and actuators are just new topics
and payloads.

## Directory layout

```
props/
├── README.md              # this file
├── ARCHITECTURE.md        # how the mesh fits together
├── PROTOCOL.md            # message format and standard topics
├── broker/                # reference Python broker
│   └── mesh_broker.py
├── lib/                   # shared client libraries
│   ├── mesh_client.py     # Python client (async)
│   └── mesh_controller.py # PirateBot integration
├── firmware/
│   ├── esp32/
│   │   └── skeleton.py    # MicroPython skeleton
│   └── rpi/
│       └── skeleton.py    # Raspberry Pi Python skeleton
├── examples/
│   ├── smoke_prop.yaml    # example prop config
│   └── thunder_prop.yaml
└── docs/
    └── wiring_guide.md
```

## Quick start

1. Start the broker on your porch PC or a dedicated Pi:
   ```bash
   cd piratebot/props/broker
   python mesh_broker.py --host 0.0.0.0 --ws-port 9001 --udp-port 9002
   ```

2. Point PirateBot at the broker in `config.portrait.yaml`:
   ```yaml
   prop_mesh:
     enabled: true
     broker_url: "ws://192.168.0.50:9001/ws"
   ```

3. Flash an ESP32 or Pi with a skeleton from `props/firmware/...`.

4. Publish a test event from anywhere:
   ```bash
   python -m props.lib.mesh_client \
     --broker ws://192.168.0.50:9001/ws \
     --publish effects.thunder.clap '{"duration_ms": 500}'
   ```

## Status

This is a clean redesign. Older prop firmware may be used for
inspiration, but new props should target the protocol in
`PROTOCOL.md`.
