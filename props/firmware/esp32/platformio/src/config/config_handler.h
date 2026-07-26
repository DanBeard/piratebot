#pragma once

#include <Arduino.h>
#include "../codec/message.h"
#include "config_manager.h"

namespace piratebot {

// Handles prop.config.* topics: load current config, apply new config,
// and request reboot. Safe: only writes to NVS, never exposes secrets.
class ConfigHandler {
public:
    bool handle(const Message& msg, Message& reply);
    bool needsReboot() const { return needs_reboot_; }
    void ackReboot() { needs_reboot_ = false; }

private:
    bool needs_reboot_ = false;

    bool handleGet(const Message& msg, Message& reply);
    bool handleSet(const Message& msg, Message& reply);
};

} // namespace piratebot
