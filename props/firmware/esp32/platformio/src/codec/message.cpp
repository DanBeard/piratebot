#include "message.h"

#include <ArduinoJson.hpp>

namespace piratebot {

static void readTiming(JsonObjectConst obj, Timing &t) {
    if (obj.containsKey("delay_ms")) {
        t.delay_ms = obj["delay_ms"];
        t.has_delay = true;
    }
    if (obj.containsKey("at_ts")) {
        t.at_ts = obj["at_ts"];
        t.has_at_ts = true;
    }
    if (obj.containsKey("expire_ms")) {
        t.expire_ms = obj["expire_ms"];
        t.has_expire = true;
    }
}

static void writeTiming(JsonObject obj, const Timing &t) {
    if (t.has_delay) obj["delay_ms"] = t.delay_ms;
    if (t.has_at_ts) obj["at_ts"] = t.at_ts;
    if (t.has_expire) obj["expire_ms"] = t.expire_ms;
}

bool Message::toJson(char *out, size_t out_len) const {
    JsonDocument doc;
    doc["topic"] = topic;
    doc["source"] = source;
    if (target[0] != '\0') doc["target"] = target;
    doc["payload"].set(payload.as<JsonObjectConst>());
    JsonObject t = doc["timing"].to<JsonObject>();
    writeTiming(t, timing);
    JsonObject m = doc["meta"].to<JsonObject>();
    m["seq"] = meta.seq;
    m["session"] = meta.session;
    JsonArray codecs = m["codecs"].to<JsonArray>();
    codecs.add("json");
    doc["timestamp"] = timestamp;

    size_t n = serializeJson(doc, out, out_len);
    return n > 0 && n < out_len;
}

bool Message::fromJson(const char *in) {
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, in);
    if (err) {
        Serial.printf("JSON parse error: %s\n", err.c_str());
        return false;
    }
    strlcpy(topic, doc["topic"] | "", sizeof(topic));
    strlcpy(source, doc["source"] | "", sizeof(source));
    strlcpy(target, doc["target"] | "", sizeof(target));
    payload.clear();
    payload.set(doc["payload"].as<JsonObjectConst>());
    readTiming(doc["timing"], timing);
    meta.seq = doc["meta"]["seq"] | 0;
    strlcpy(meta.session, doc["meta"]["session"] | MESH_SESSION, sizeof(meta.session));
    timestamp = doc["timestamp"] | 0.0;
    return true;
}

bool Message::toCbor(uint8_t *out, size_t &out_len) const {
    // CBOR support is not compiled in by default because no common ESP32
    // Arduino library exposes it. Use framed JSON or add a CBOR library.
    (void)out;
    (void)out_len;
    Serial.println("CBOR not implemented on ESP32; use JSON or add a CBOR library");
    return false;
}

bool Message::fromCbor(const uint8_t *in, size_t in_len) {
    (void)in;
    (void)in_len;
    Serial.println("CBOR not implemented on ESP32; use JSON or add a CBOR library");
    return false;
}

bool Message::toFramed(uint8_t *out, size_t &out_len, uint8_t codec, uint8_t flags) const {
    if (codec != MESH_CODEC_JSON) return false;

    char json_buf[MESH_JSON_BUFFER_SIZE];
    if (!toJson(json_buf, sizeof(json_buf))) return false;
    size_t payload_len = strlen(json_buf);
    if (payload_len > MESH_FRAMED_BUFFER_SIZE) return false;

    if (payload_len > 0xFFFF) return false;

    out[0] = MESH_MAGIC_0;
    out[1] = MESH_MAGIC_1;
    out[2] = MESH_VERSION;
    out[3] = flags;
    out[4] = codec;
    out[5] = (payload_len >> 8) & 0xFF;
    out[6] = payload_len & 0xFF;
    memcpy(out + 7, json_buf, payload_len);
    out_len = 7 + payload_len;
    return true;
}

bool Message::fromFramed(const uint8_t *in, size_t in_len) {
    if (in_len < 7) return false;
    if (in[0] != MESH_MAGIC_0 || in[1] != MESH_MAGIC_1) return false;
    if (in[2] != MESH_VERSION) return false;
    uint8_t codec = in[4];
    size_t payload_len = ((size_t)in[5] << 8) | in[6];
    if (in_len < 7 + payload_len) return false;

    if (codec == MESH_CODEC_JSON) {
        char json_buf[MESH_JSON_BUFFER_SIZE];
        if (payload_len >= sizeof(json_buf)) return false;
        memcpy(json_buf, in + 7, payload_len);
        json_buf[payload_len] = '\0';
        return fromJson(json_buf);
    }
    if (codec == MESH_CODEC_CBOR) {
        return fromCbor(in + 7, payload_len);
    }
    return false;
}

} // namespace piratebot
