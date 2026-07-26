#pragma once

#include <Arduino.h>
#include "../codec/message.h"

namespace piratebot {

// HTTP OTA updater. Disabled by default. Must be explicitly enabled via
// prop.ota.enable with confirm=true and only when the show scene is idle.
class OtaManager {
public:
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    // Handle prop.ota.* topics. Returns true if the topic was consumed.
    bool handle(const Message& msg, Message& reply, const char* current_scene);

    // Returns true if an update was requested and completed; caller should reboot.
    bool needsReboot() const { return needs_reboot_; }

private:
    bool enabled_ = false;
    bool needs_reboot_ = false;

    bool handleEnable(const Message& msg, Message& reply, const char* current_scene);
    bool handleStart(const Message& msg, Message& reply, const char* current_scene);
    bool isSceneSafe(const char* scene) const;
};

} // namespace piratebot
