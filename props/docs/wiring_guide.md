# Prop Wiring Guide

## Power

- ESP32: 5V via micro-USB or a 5V buck converter from a 12V prop supply.
- Pi Zero: 5V 2A minimum; use a quality cable. Power-hungry relays need
  separate power, not the Pi's 5V rail.

## Relays and safety

- Use opto-isolated relay modules for AC fog machines and strobes.
- Keep mains wiring in enclosures; don't run 120V/240V on breadboards.
- Fuse everything. Fog machines pull serious current on startup.

## Timing-critical combos

If a prop needs thunder + strobe + light locked to within a few
milliseconds, do it on the same board. Use the mesh only for coarse
triggers or to start a local pre-programmed sequence.

Example: the thunder Pi receives `effects.thunder.clap` and locally
plays WAV audio while toggling a strobe and LED strip. The network does
not carry the individual flashes; the Pi does.

## Network

- Put all props on the same dedicated Halloween AP or VLAN if possible.
- Reserve DHCP addresses or use mDNS names so broker URLs are stable.
- Test with `ping -f` from the broker host to verify WiFi latency.

## Sensors

- PIR: simple digital input; debounce in firmware.
- Pressure mat: switch or force-sensitive resistor.
- Beam break: IR LED + phototransistor pair.

Publish sensor events with a `zone` field so PirateBot can reason about
where the kid is.
