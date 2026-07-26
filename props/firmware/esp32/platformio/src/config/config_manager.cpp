#include "config_manager.h"

#include <Preferences.h>

namespace piratebot {

static Preferences prefs;

bool ConfigManager::begin() {
    opened_ = prefs.begin("piratebot", false);
    return opened_;
}

void ConfigManager::load(PropConfig& out) {
    if (!opened_) begin();
    char buf[128];
    if (prefs.getString("node_id", buf, sizeof(buf))) strlcpy(out.node_id, buf, sizeof(out.node_id));
    if (prefs.getString("profile", buf, sizeof(buf))) strlcpy(out.profile, buf, sizeof(out.profile));
    if (prefs.getString("wifi_ssid", buf, sizeof(buf))) strlcpy(out.wifi_ssid, buf, sizeof(out.wifi_ssid));
    if (prefs.getString("wifi_pass", buf, sizeof(buf))) strlcpy(out.wifi_pass, buf, sizeof(out.wifi_pass));
    if (prefs.getString("broker_host", buf, sizeof(buf))) strlcpy(out.broker_host, buf, sizeof(out.broker_host));
    out.broker_ws_port = prefs.getUShort("broker_ws_port", out.broker_ws_port);
    out.broker_mqtt_port = prefs.getUShort("broker_mqtt_port", out.broker_mqtt_port);
    out.ota_enabled = prefs.getBool("ota_enabled", out.ota_enabled);
}

bool ConfigManager::save(const PropConfig& cfg) {
    if (!opened_) begin();
    prefs.putString("node_id", cfg.node_id);
    prefs.putString("profile", cfg.profile);
    prefs.putString("wifi_ssid", cfg.wifi_ssid);
    prefs.putString("wifi_pass", cfg.wifi_pass);
    prefs.putString("broker_host", cfg.broker_host);
    prefs.putUShort("broker_ws_port", cfg.broker_ws_port);
    prefs.putUShort("broker_mqtt_port", cfg.broker_mqtt_port);
    prefs.putBool("ota_enabled", cfg.ota_enabled);
    return true;
}

bool ConfigManager::set(const char* key, const char* value) {
    if (!opened_) begin();
    return prefs.putString(key, value) > 0;
}

bool ConfigManager::set(const char* key, uint32_t value) {
    if (!opened_) begin();
    return prefs.putUInt(key, value) == 4;
}

bool ConfigManager::get(const char* key, char* out, size_t out_len) {
    if (!opened_) begin();
    return prefs.getString(key, out, out_len) > 0;
}

bool ConfigManager::get(const char* key, uint32_t& out) {
    if (!opened_) begin();
    out = prefs.getUInt(key, 0);
    return true;
}

void ConfigManager::clear() {
    if (!opened_) begin();
    prefs.clear();
}

} // namespace piratebot
