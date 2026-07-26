#include <Arduino.h>
#include <WiFi.h>
#include <Preferences.h>

#include "config.h"
#include "codec/message.h"
#include "effect_engine.h"
#include "sensor_engine.h"
#include "profiles.h"
#include "transports/ws_client.h"
#include "transports/mqtt_client.h"
#include "transports/discovery.h"
#include "config/config_manager.h"
#include "config/config_handler.h"
#include "ota/ota_manager.h"

using namespace piratebot;

static EffectEngine effect_engine;
static SensorEngine sensor_engine([](const Message& msg) {
    // Sensors send via all available transports.
    ws_client.send(msg);
    mqtt_client.send(msg);
});

static WebSocketClient ws_client;
static MqttClient mqtt_client;
static DiscoveryClient discovery;
static ConfigManager config_mgr;
static ConfigHandler config_handler;
static OtaManager ota_mgr;
static char current_scene[32] = "idle";

static uint32_t last_heartbeat_ms = 0;
static uint32_t last_reconnect_ms = 0;
static uint32_t seq = 0;

static void on_mesh_message(const Message& msg) {
    if (msg.target[0] != '\0' && strcmp(msg.target, MESH_NODE_ID) != 0) return;

    if (strcmp(msg.topic, "scene.estop") == 0) {
        strlcpy(current_scene, "estop", sizeof(current_scene));
        effect_engine.stopAll();
        return;
    }
    if (strcmp(msg.topic, "scene.resume") == 0) {
        strlcpy(current_scene, "idle", sizeof(current_scene));
        return;
    }
    if (strncmp(msg.topic, "scene.", 6) == 0) {
        const char* new_scene = msg.payload["scene"] | "idle";
        strlcpy(current_scene, new_scene, sizeof(current_scene));
        return;
    }

    Message reply;
    if (config_handler.handle(msg, reply)) {
        ws_client.send(reply);
        mqtt_client.send(reply);
        if (config_handler.needsReboot()) {
            delay(500);
            ESP.restart();
        }
        return;
    }

    if (ota_mgr.handle(msg, reply, current_scene)) {
        ws_client.send(reply);
        mqtt_client.send(reply);
        if (ota_mgr.needsReboot()) {
            delay(500);
            ESP.restart();
        }
        return;
    }

    effect_engine.handleMessage(msg);
}

static void loadProfile() {
    PropConfig cfg;
    config_mgr.begin();
    config_mgr.load(cfg);
    config_mgr.end();

    char profile[64];
    strlcpy(profile, cfg.profile, sizeof(profile));
    if (profile[0] == '\0') {
        strncpy(profile, "cannon", sizeof(profile));
    }

    Serial.printf("Loading profile: %s\n", profile);
    if (strcmp(profile, "cannon") == 0) effect_engine.addProfile(&CANNON_PROFILE);
    else if (strcmp(profile, "smoke") == 0) effect_engine.addProfile(&SMOKE_PROFILE);
    else if (strcmp(profile, "strobe") == 0) effect_engine.addProfile(&STROBE_PROFILE);
    else if (strcmp(profile, "thunder") == 0) effect_engine.addProfile(&THUNDER_PROFILE);
    else if (strcmp(profile, "relay") == 0) effect_engine.addProfile(&RELAY_PROFILE);
    else if (strcmp(profile, "pir") == 0) sensor_engine.addProfile(&PIR_PROFILE);
    else if (strcmp(profile, "beam") == 0) sensor_engine.addProfile(&BEAM_PROFILE);
    else {
        // Multi-profile fallback: comma-separated list in NVS.
        effect_engine.addProfile(&CANNON_PROFILE);
        effect_engine.addProfile(&SMOKE_PROFILE);
        sensor_engine.addProfile(&PIR_PROFILE);
    }

    ota_mgr.setEnabled(cfg.ota_enabled);
}

static void connectNetwork() {
    PropConfig cfg;
    config_mgr.begin();
    config_mgr.load(cfg);
    config_mgr.end();

    WiFi.mode(WIFI_STA);
    WiFi.begin(cfg.wifi_ssid, cfg.wifi_pass);
    Serial.print("Connecting to WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(250);
        Serial.print(".");
        ++attempts;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nWiFi connected: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWiFi connect failed");
    }
}

static void discoverBroker() {
    PropConfig cfg;
    config_mgr.begin();
    config_mgr.load(cfg);
    config_mgr.end();

    char broker_host[64];
    strlcpy(broker_host, cfg.broker_host[0] != '\0' ? cfg.broker_host : BROKER_HOST, sizeof(broker_host));
    uint16_t broker_port = cfg.broker_ws_port ? cfg.broker_ws_port : BROKER_WS_PORT;

    if (broker_host[0] == '\0') {
        Serial.println("Discovering broker via UDP multicast...");
        if (discovery.findBroker(broker_host, sizeof(broker_host), broker_port, 5000)) {
            Serial.printf("Found broker at %s:%u\n", broker_host, broker_port);
        } else {
            Serial.println("Broker discovery failed; falling back");
            strncpy(broker_host, "192.168.0.50", sizeof(broker_host));
        }
    }

    ws_client.begin(broker_host, broker_port, MESH_NODE_ID);
    ws_client.onMessage(on_mesh_message);
}

static void announce() {
    Message msg;
    strlcpy(msg.topic, "prop.state.announce", sizeof(msg.topic));
    strlcpy(msg.source, MESH_NODE_ID, sizeof(msg.source));
    msg.payload["id"] = MESH_NODE_ID;
    msg.payload["name"] = "PirateBot ESP32 Prop";
    msg.payload["transport"] = "websocket";

    char* caps[8];
    uint8_t cap_count = 0;
    effect_engine.getCapabilities(caps, cap_count, 8);
    JsonArray arr = msg.payload.createNestedArray("capabilities");
    for (uint8_t i = 0; i < cap_count; ++i) arr.add(caps[i]);
    arr.add("prop.config.get");
    arr.add("prop.config.set");
    arr.add("prop.ota.enable");

    JsonArray codecs = msg.payload.createNestedArray("codecs");
    codecs.add("json");
    codecs.add("cbor");  // advertised even if not yet implemented

    msg.meta.seq = ++seq;
    msg.timestamp = millis() / 1000.0;

    ws_client.send(msg);
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("PirateBot ESP32 prop starting");

    loadProfile();
    connectNetwork();
    discoverBroker();

    mqtt_client.begin("192.168.0.2", 1883, MESH_NODE_ID);
    mqtt_client.onMessage(on_mesh_message);

    ws_client.onMessage(on_mesh_message);

    delay(500);
    announce();
}

void loop() {
    ws_client.update();
    mqtt_client.update();
    effect_engine.update();
    sensor_engine.update();

    uint32_t now = millis();
    if (now - last_heartbeat_ms > 5000) {
        last_heartbeat_ms = now;
        Message hb;
        strlcpy(hb.topic, "prop.state.heartbeat", sizeof(hb.topic));
        strlcpy(hb.source, MESH_NODE_ID, sizeof(hb.source));
        hb.payload["uptime_s"] = now / 1000;
        hb.payload["rssi"] = WiFi.RSSI();
        hb.meta.seq = ++seq;
        hb.timestamp = now / 1000.0;
        ws_client.send(hb);
    }

    if (now - last_reconnect_ms > 30000) {
        last_reconnect_ms = now;
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.reconnect();
        }
        if (!ws_client.connected()) {
            announce();
        }
    }

    delay(5);
}
