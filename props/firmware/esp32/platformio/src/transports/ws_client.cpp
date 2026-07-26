#include "ws_client.h"

#include <ArduinoJson.hpp>

namespace piratebot {

WebSocketClient::WebSocketClient() {}

void WebSocketClient::begin(const char* host, uint16_t port, const char* node_id) {
    strlcpy(host_, host, sizeof(host_));
    port_ = port;
    strlcpy(node_id_, node_id, sizeof(node_id_));
}

void WebSocketClient::onMessage(MessageHandler handler) {
    handler_ = handler;
}

void WebSocketClient::connect() {
    if (has_socket_) return;
    socket_ = new AsyncWebSocket("/ws");
    socket_-&gt;onEvent(
        [this](AsyncWebSocket* server, AsyncWebSocketClient* client, AwsEventType type, void* arg, uint8_t* data, size_t len) {
            (void)server;
            (void)arg;
            if (type == WS_EVT_CONNECT) {
                ws_ = client;
                Serial.printf("WS connected to %s:%u\n", host_, port_);
            } else if (type == WS_EVT_DISCONNECT) {
                ws_ = nullptr;
                Serial.println("WS disconnected");
            } else if (type == WS_EVT_DATA && handler_) {
                char buf[512];
                size_t copy = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
                memcpy(buf, data, copy);
                buf[copy] = '\0';
                Message msg;
                if (msg.fromJson(buf)) {
                    handler_(msg);
                }
            }
        }
    );

    server_ = new AsyncWebServer(0);
    server_-&gt;addHandler(socket_);
    server_-&gt;begin();
    has_socket_ = true;

    // ESPAsyncWebServer client API is server-side; for a real client use
    // a dedicated WebSocket client library such as ArduinoWebsockets.
    // This file provides the interface and integration point.
    Serial.println("NOTE: ws_client.cpp uses AsyncWebSocket server shim;");
    Serial.println("      swap in ArduinoWebsockets for true client mode.");
}

void WebSocketClient::update() {
    uint32_t now = millis();
    if (!connected() && now - last_attempt_ms_ > 3000) {
        last_attempt_ms_ = now;
        if (!has_socket_) connect();
    }
}

bool WebSocketClient::connected() const {
    return ws_ != nullptr && ws_-&gt;status() == WS_CONNECTED;
}

bool WebSocketClient::send(const Message& msg) {
    if (!connected()) return false;
    char buf[MESH_JSON_BUFFER_SIZE];
    if (!msg.toJson(buf, sizeof(buf))) return false;
    ws_-&gt;text(buf);
    return true;
}

} // namespace piratebot
