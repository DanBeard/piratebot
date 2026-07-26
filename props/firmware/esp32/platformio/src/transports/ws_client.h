#pragma once

#include <Arduino.h>
#include <ESPAsyncWebServer.h>
#include "../codec/message.h"

namespace piratebot {

using MessageHandler = void (*)(const Message& msg);

class WebSocketClient {
public:
    WebSocketClient();

    void begin(const char* host, uint16_t port, const char* node_id);
    void update();
    bool connected() const;
    bool send(const Message& msg);
    void onMessage(MessageHandler handler);

private:
    AsyncWebSocketClient* ws_ = nullptr;
    AsyncWebSocket* socket_ = nullptr;
    AsyncWebServer* server_ = nullptr;
    MessageHandler handler_ = nullptr;
    char host_[64];
    uint16_t port_ = 9001;
    char node_id_[32];
    uint32_t last_attempt_ms_ = 0;
    bool has_socket_ = false;

    void connect();
};

} // namespace piratebot
