#pragma once

#include <Arduino.h>

// Default credentials and broker discovery are read from NVS or fall back
// to compile-time defaults. Override with your own build flags or by flashing
// a config.json to LittleFS.

#ifndef MESH_SESSION
#define MESH_SESSION "halloween-2026"
#endif

#ifndef MESH_NODE_ID
#define MESH_NODE_ID "prop_esp32"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASS
#define WIFI_PASS ""
#endif

#ifndef BROKER_HOST
#define BROKER_HOST ""
#endif

#ifndef BROKER_WS_PORT
#define BROKER_WS_PORT 9001
#endif

#ifndef BROKER_UDP_PORT
#define BROKER_UDP_PORT 9002
#endif

#ifndef DISCOVERY_MCAST_GRP
#define DISCOVERY_MCAST_GRP "239.255.42.99"
#endif

constexpr uint16_t MESH_JSON_BUFFER_SIZE = 1024;
constexpr uint16_t MESH_FRAMED_BUFFER_SIZE = 768;
constexpr uint8_t MESH_MAGIC_0 = 0x50;  // 'P'
constexpr uint8_t MESH_MAGIC_1 = 0x42;  // 'B'
constexpr uint8_t MESH_VERSION = 0x01;

// Flags bits
constexpr uint8_t MESH_FLAG_FLOOD_MESH = 0x01;
constexpr uint8_t MESH_FLAG_TLV_BINARY = 0x02;

// Codec bytes
constexpr uint8_t MESH_CODEC_JSON = 0x01;
constexpr uint8_t MESH_CODEC_CBOR = 0x02;

// Default safety limits
constexpr uint32_t DEFAULT_MAX_ON_MS = 5000;
constexpr uint32_t DEFAULT_COOLDOWN_MS = 1000;
