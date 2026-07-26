#include "ota_manager.h"

#ifndef UNIT_TEST
#include <Update.h>
#include <HTTPClient.h>
#include <WiFi.h>
#endif

namespace piratebot {

bool OtaManager::isSceneSafe(const char* scene) const {
    if (!scene) return false;
    return strcmp(scene, "idle") == 0 || strcmp(scene, "quiet") == 0 || strcmp(scene, "estop") == 0;
}

bool OtaManager::handle(const Message& msg, Message& reply, const char* current_scene) {
    if (strcmp(msg.topic, "prop.ota.enable") == 0) {
        return handleEnable(msg, reply, current_scene);
    }
    if (strcmp(msg.topic, "prop.ota.start") == 0) {
        return handleStart(msg, reply, current_scene);
    }
    return false;
}

bool OtaManager::handleEnable(const Message& msg, Message& reply, const char* current_scene) {
    bool confirm = msg.payload["confirm"] | false;
    bool want = msg.payload["enabled"] | false;

    strlcpy(reply.topic, "prop.ota.status", sizeof(reply.topic));
    strlcpy(reply.source, MESH_NODE_ID, sizeof(reply.source));
    if (msg.source[0] != '\0') strlcpy(reply.target, msg.source, sizeof(reply.target));

    if (!confirm) {
        reply.payload["error"] = "confirm:true required to enable OTA";
        reply.payload["enabled"] = enabled_;
        return true;
    }

    enabled_ = want;
    reply.payload["enabled"] = enabled_;
    reply.payload["safe_scene"] = isSceneSafe(current_scene);
    return true;
}

bool OtaManager::handleStart(const Message& msg, Message& reply, const char* current_scene) {
    strlcpy(reply.topic, "prop.ota.status", sizeof(reply.topic));
    strlcpy(reply.source, MESH_NODE_ID, sizeof(reply.source));
    if (msg.source[0] != '\0') strlcpy(reply.target, msg.source, sizeof(reply.target));

    if (!enabled_) {
        reply.payload["error"] = "OTA is disabled; send prop.ota.enable with confirm:true first";
        reply.payload["enabled"] = false;
        return true;
    }
    if (!isSceneSafe(current_scene)) {
        reply.payload["error"] = "OTA only allowed in idle/quiet/estop scene";
        reply.payload["scene"] = current_scene;
        return true;
    }

    const char* url = msg.payload["url"];
    if (!url || strlen(url) == 0) {
        reply.payload["error"] = "missing url";
        return true;
    }

    #ifdef UNIT_TEST
    // In native tests we cannot actually flash hardware. Pretend success.
    reply.payload["success"] = true;
    reply.payload["bytes_written"] = 0;
    needs_reboot_ = true;
    return true;
    #else
    HTTPClient http;
    http.begin(url);
    int httpCode = http.GET();
    if (httpCode != 200) {
        reply.payload["error"] = "http failed";
        reply.payload["http_code"] = httpCode;
        http.end();
        return true;
    }

    int contentLength = http.getSize();
    if (contentLength <= 0) {
        reply.payload["error"] = "invalid content length";
        http.end();
        return true;
    }

    WiFiClient* stream = http.getStreamPtr();
    if (!Update.begin(contentLength)) {
        reply.payload["error"] = "Update.begin failed";
        http.end();
        return true;
    }

    size_t written = 0;
    uint8_t buf[512];
    while (http.connected() && written < (size_t)contentLength) {
        size_t available = stream->available();
        if (available) {
            size_t toRead = available > sizeof(buf) ? sizeof(buf) : available;
            size_t n = stream->readBytes(buf, toRead);
            if (n > 0) {
                written += Update.write(buf, n);
            }
        }
        delay(1);
    }

    http.end();

    bool success = Update.end() && written == (size_t)contentLength;
    reply.payload["success"] = success;
    reply.payload["bytes_written"] = written;
    if (!success) {
        reply.payload["error"] = "update incomplete";
    } else {
        needs_reboot_ = true;
    }
    return true;
    #endif
}

} // namespace piratebot
