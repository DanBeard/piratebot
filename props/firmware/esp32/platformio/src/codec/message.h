#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include "../config.h"

namespace piratebot {

// Timing fields are optional; zero means not set.
struct Timing {
    int32_t delay_ms = 0;
    double at_ts = 0.0;
    int32_t expire_ms = 0;
    bool has_delay = false;
    bool has_at_ts = false;
    bool has_expire = false;
};

struct Meta {
    uint32_t seq = 0;
    char session[32] = MESH_SESSION;
    bool cbor_supported = true;
};

struct Message {
    char topic[64] = "";
    char source[32] = MESH_NODE_ID;
    char target[32] = "";
    StaticJsonDocument<MESH_JSON_BUFFER_SIZE> payload;
    Timing timing;
    Meta meta;
    double timestamp = 0.0;

    // Serializers
    bool toJson(char* out, size_t out_len) const;
    bool toCbor(uint8_t* out, size_t& out_len) const;
    bool toFramed(uint8_t* out, size_t& out_len, uint8_t codec = MESH_CODEC_CBOR, uint8_t flags = 0) const;

    // Deserializers
    bool fromJson(const char* in);
    bool fromCbor(const uint8_t* in, size_t in_len);
    bool fromFramed(const uint8_t* in, size_t in_len);
};

}  // namespace piratebot
