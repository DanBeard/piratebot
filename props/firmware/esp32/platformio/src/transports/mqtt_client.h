#pragma once

#include <Arduino.h>
#include <AsyncMqttClient.h>
#include "../codec/message.h"

namespace piratebot {

using MessageHandler = void (*)(const Message& msg);

class MqttClient {
public:
    MqttClient();

    void begin(const char* broker_host, uint16_t broker_port, const char* node_id);
    void update();
    bool connected() const;
    bool send(const Message& msg);
    void onMessage(MessageHandler handler);

private:
    AsyncMqttClient mqtt_;
    MessageHandler handler_ = nullptr;
    char host_[64];
    uint16_t port_ = 1883;
    char node_id_[32];
    char pub_topic_[64];
    char sub_topic_[64];
    bool connecting_ = false;
    uint32_t last_attempt_ms_ = 0;

    void onConnect(bool sessionPresent);
    void onDisconnect(AsyncMqttClientDisconnectReason reason);
    void onMessageReceived(char* topic, char* payload, AsyncMqttClientMessageProperties props, size_t len, size_t index, size_t total);
};

} // namespace piratebot
