#pragma once

#include <Arduino.h>
#include "../config.h"

namespace piratebot {

// Keys stored in NVS namespace "piratebot"
struct PropConfig {
    char node_id[32] = MESH_NODE_ID;
    char profile[64] = "";
    char wifi_ssid[64] = WIFI_SSID;
    char wifi_pass[64] = WIFI_PASS;
    char broker_host[64] = BROKER_HOST;
    uint16_t broker_ws_port = BROKER_WS_PORT;
    uint16_t broker_mqtt_port = 1883;
    bool ota_enabled = false;
};

class ConfigManager {
public:
    bool begin();
    void load(PropConfig& out);
    bool save(const PropConfig& cfg);
    bool set(const char* key, const char* value);
    bool set(const char* key, uint32_t value);
    bool get(const char* key, char* out, size_t out_len);
    bool get(const char* key, uint32_t& out);
    void clear();

private:
    bool opened_ = false;
};

} // namespace piratebot
