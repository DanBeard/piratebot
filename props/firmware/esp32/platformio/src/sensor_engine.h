#pragma once

#include <Arduino.h>
#include "profiles.h"
#include "codec/message.h"

namespace piratebot {

using SensorCallback = void (*)(const Message& msg);

// Debounced sensor reader that emits mesh messages on state changes.
class SensorEngine {
public:
    SensorEngine(SensorCallback cb);

    void addProfile(const SensorProfile* profile);
    void update();

private:
    static constexpr uint8_t MAX_PROFILES = 4;
    const SensorProfile* profiles_[MAX_PROFILES];
    uint8_t profile_count_ = 0;

    uint8_t last_state_[MAX_PROFILES];
    uint32_t last_change_ms_[MAX_PROFILES];
    uint32_t last_emit_ms_[MAX_PROFILES];
    uint32_t seq_ = 0;

    SensorCallback callback_;

    void emit(const SensorProfile* profile, bool detected);
};

} // namespace piratebot
