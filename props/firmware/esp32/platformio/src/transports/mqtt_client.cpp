#include "mqtt_client.h"

namespace piratebot {

static MqttClient* instance = nullptr;

MqttClient::MqttClient() {
    instance = this;
    mqtt_.onConnect([](bool sessionPresent) { if (instance) instance-&gt;onConnect(sessionPresent); });
    mqtt_.onDisconnect([](AsyncMqttClientDisconnectReason reason) { if (instance) instance-&gt;onDisconnect(reason); });
    mqtt_.onMessage([](char* topic, char* payload, AsyncMqttClientMessageProperties props, size_t len, size_t index, size_t total) {
        if (instance) instance-&gt;onMessageReceived(topic, payload, props, len, index, total);
    });
}

void MqttClient::begin(const char* broker_host, uint16_t broker_port, const char* node_id) {
    strlcpy(host_, broker_host, sizeof(host_));
    port_ = broker_port;
    strlcpy(node_id_, node_id, sizeof(node_id_));
    snprintf(pub_topic_, sizeof(pub_topic_), "piratebot/%s/out", node_id_);
    snprintf(sub_topic_, sizeof(sub_topic_), "piratebot/%s/in", node_id_);
}

void MqttClient::onMessage(MessageHandler handler) {
    handler_ = handler;
}

void MqttClient::update() {
    uint32_t now = millis();
    if (!connected() && !connecting_ && now - last_attempt_ms_ > 3000) {
        connecting_ = true;
        last_attempt_ms_ = now;
        mqtt_.setServer(host_, port_);
        mqtt_.connect();
    }
}

bool MqttClient::connected() const {
    return mqtt_.connected();
}

void MqttClient::onConnect(bool sessionPresent) {
    (void)sessionPresent;
    connecting_ = false;
    Serial.println("MQTT connected");
    mqtt_.subscribe(sub_topic_, 0);
}

void MqttClient::onDisconnect(AsyncMqttClientDisconnectReason reason) {
    (void)reason;
    connecting_ = false;
    Serial.println("MQTT disconnected");
}

void MqttClient::onMessageReceived(char* topic, char* payload, AsyncMqttClientMessageProperties props, size_t len, size_t index, size_t total) {
    (void)topic;
    (void)props;
    (void)index;
    (void)total;
    if (!handler_) return;
    char buf[MESH_JSON_BUFFER_SIZE];
    size_t copy = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, payload, copy);
    buf[copy] = '\0';
    Message msg;
    if (msg.fromJson(buf)) {
        handler_(msg);
    }
}

bool MqttClient::send(const Message& msg) {
    if (!connected()) return false;
    char buf[MESH_JSON_BUFFER_SIZE];
    if (!msg.toJson(buf, sizeof(buf))) return false;
    mqtt_.publish(pub_topic_, 0, false, buf);
    return true;
}

} // namespace piratebot
