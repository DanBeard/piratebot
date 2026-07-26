#include "config_handler.h"

namespace piratebot {

bool ConfigHandler::handle(const Message& msg, Message& reply) {
    if (strcmp(msg.topic, "prop.config.get") == 0) {
        return handleGet(msg, reply);
    }
    if (strcmp(msg.topic, "prop.config.set") == 0) {
        return handleSet(msg, reply);
    }
    return false;
}

bool ConfigHandler::handleGet(const Message& msg, Message& reply) {
    ConfigManager mgr;
    PropConfig cfg;
    mgr.load(cfg);

    strlcpy(reply.topic, "prop.config.current", sizeof(reply.topic));
    strlcpy(reply.source, MESH_NODE_ID, sizeof(reply.source));
    if (msg.source[0] != '\0') strlcpy(reply.target, msg.source, sizeof(reply.target));
    reply.payload["node_id"] = cfg.node_id;
    reply.payload["profile"] = cfg.profile;
    reply.payload["wifi_ssid"] = cfg.wifi_ssid;
    reply.payload["broker_host"] = cfg.broker_host;
    reply.payload["broker_ws_port"] = cfg.broker_ws_port;
    reply.payload["broker_mqtt_port"] = cfg.broker_mqtt_port;
    reply.payload["ota_enabled"] = cfg.ota_enabled;
    // Intentionally do NOT send wifi_pass or broker secrets back.
    return true;
}

bool ConfigHandler::handleSet(const Message& msg, Message& reply) {
    ConfigManager mgr;
    PropConfig cfg;
    mgr.load(cfg);

    if (msg.payload.containsKey("node_id")) {
        strlcpy(cfg.node_id, msg.payload["node_id"], sizeof(cfg.node_id));
    }
    if (msg.payload.containsKey("profile")) {
        strlcpy(cfg.profile, msg.payload["profile"], sizeof(cfg.profile));
    }
    if (msg.payload.containsKey("wifi_ssid")) {
        strlcpy(cfg.wifi_ssid, msg.payload["wifi_ssid"], sizeof(cfg.wifi_ssid));
    }
    if (msg.payload.containsKey("wifi_pass")) {
        strlcpy(cfg.wifi_pass, msg.payload["wifi_pass"], sizeof(cfg.wifi_pass));
    }
    if (msg.payload.containsKey("broker_host")) {
        strlcpy(cfg.broker_host, msg.payload["broker_host"], sizeof(cfg.broker_host));
    }
    if (msg.payload.containsKey("broker_ws_port")) {
        cfg.broker_ws_port = msg.payload["broker_ws_port"];
    }
    if (msg.payload.containsKey("broker_mqtt_port")) {
        cfg.broker_mqtt_port = msg.payload["broker_mqtt_port"];
    }
    if (msg.payload.containsKey("ota_enabled")) {
        cfg.ota_enabled = msg.payload["ota_enabled"];
    }

    bool ok = mgr.save(cfg);

    strlcpy(reply.topic, "prop.config.ack", sizeof(reply.topic));
    strlcpy(reply.source, MESH_NODE_ID, sizeof(reply.source));
    if (msg.source[0] != '\0') strlcpy(reply.target, msg.source, sizeof(reply.target));
    reply.payload["success"] = ok;
    reply.payload["needs_reboot"] = true;

    needs_reboot_ = true;
    return true;
}

} // namespace piratebot
