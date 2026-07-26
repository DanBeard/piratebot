"""
MicroPython skeleton for an ESP32 prop node.

This file is a starting point. Copy it to your ESP32 as main.py and fill
in the effect code for your specific prop.
"""

from __future__ import annotations

import json
import time
import network
import uasyncio as asyncio
from machine import Pin

try:
    import uwebsockets.client as ws
except ImportError:
    ws = None  # type: ignore

WIFI_SSID = "your-ssid"
WIFI_PASS = "your-pass"
MESH_BROKER = "ws://192.168.0.50:9001/ws"
PROP_ID = "esp32_prop_01"
SESSION = "halloween-2026"

# Example effect pins
LED_PIN = 2
RELAY_PIN = 4

wlan = network.WLAN(network.STA_IF)
led = Pin(LED_PIN, Pin.OUT)
relay = Pin(RELAY_PIN, Pin.OUT)


async def connect_wifi():
    wlan.active(True)
    if wlan.isconnected():
        return
    wlan.connect(WIFI_SSID, WIFI_PASS)
    while not wlan.isconnected():
        await asyncio.sleep(1)
    print("WiFi connected:", wlan.ifconfig()[0])


async def discover_broker():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    req = json.dumps({"cmd": "discover", "session": SESSION}).encode()
    sock.sendto(req, ("239.255.42.99", 9002))
    try:
        data, _ = sock.recvfrom(1024)
        reply = json.loads(data.decode())
        return reply.get("broker")
    except Exception:
        return MESH_BROKER


async def heartbeat(ws_client):
    while True:
        msg = {
            "topic": "prop.state.heartbeat",
            "source": PROP_ID,
            "target": None,
            "payload": {"uptime_s": time.time()},
            "timing": {},
            "meta": {"session": SESSION},
            "timestamp": time.time(),
        }
        try:
            ws_client.send(json.dumps(msg))
        except Exception as exc:
            print("heartbeat failed:", exc)
            return
        await asyncio.sleep(30)


def handle_message(data):
    topic = data.get("topic")
    payload = data.get("payload", {})
    timing = data.get("timing", {})

    delay_ms = timing.get("delay_ms", 0)
    if delay_ms:
        time.sleep(delay_ms / 1000.0)

    if topic == "effects.strobe.flash":
        duration_ms = payload.get("duration_ms", 500)
        flash(duration_ms)
    elif topic == "effects.smoke.burst":
        duration_ms = payload.get("duration_ms", 2000)
        burst_smoke(duration_ms)
    elif topic == "effects.thunder.clap":
        duration_ms = payload.get("duration_ms", 800)
        thunder(duration_ms)
    else:
        print("unhandled topic:", topic)


def flash(duration_ms):
    print("strobe", duration_ms)
    led.on()
    time.sleep_ms(duration_ms)
    led.off()


def burst_smoke(duration_ms):
    print("smoke", duration_ms)
    relay.on()
    time.sleep_ms(duration_ms)
    relay.off()


def thunder(duration_ms):
    print("thunder", duration_ms)
    # Add sound trigger here
    flash(duration_ms)


async def main():
    await connect_wifi()
    broker = await discover_broker()
    print("broker:", broker)

    if ws is None:
        raise RuntimeError("uwebsockets not available")

    ws_client = ws.connect(broker)

    # Announce capabilities
    announce = {
        "topic": "prop.state.announce",
        "source": PROP_ID,
        "target": None,
        "payload": {
            "capabilities": ["effects.strobe.flash", "effects.smoke.burst", "effects.thunder.clap"]
        },
        "timing": {},
        "meta": {"session": SESSION},
        "timestamp": time.time(),
    }
    ws_client.send(json.dumps(announce))

    asyncio.create_task(heartbeat(ws_client))

    while True:
        try:
            raw = ws_client.recv()
            if raw:
                data = json.loads(raw)
                handle_message(data)
        except Exception as exc:
            print("receive error:", exc)
            await asyncio.sleep(2)


asyncio.run(main())
