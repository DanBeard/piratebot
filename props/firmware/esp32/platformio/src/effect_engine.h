#pragma once

#include <Arduino.h>
#include "profiles.h"
#include "codec/message.h"

namespace piratebot {

// Run timed actuator effects with safety limits and cooldowns.
class EffectEngine {
public:
    EffectEngine();

    // Add an effect profile to the active set. Returns capabilities list.
    void addProfile(const EffectProfile* profile);

    // Try to start an effect from a mesh message. Returns true if fired.
    bool handleMessage(const Message& msg);

    // Must be called frequently from loop().
    void update();

    // Immediately stop all effects (e.g., E-Stop).
    void stopAll();

    // Fill capabilities array for announce payload.
    void getCapabilities(char** out, uint8_t& count, uint8_t max) const;

private:
    static constexpr uint8_t MAX_PROFILES = 8;
    const EffectProfile* profiles_[MAX_PROFILES];
    uint8_t profile_count_ = 0;

    uint32_t last_fire_ms_[MAX_PROFILES];
    bool active_[MAX_PROFILES];
    uint32_t segment_start_ms_[MAX_PROFILES];
    int32_t delay_ms_[MAX_PROFILES];
    uint8_t segment_index_[MAX_PROFILES];
    uint8_t repeat_count_[MAX_PROFILES];

    int findProfileIndex(const char* topic) const;
    void setPin(const GpioSegment& seg, bool on);
    bool canFire(uint8_t idx);
    void complete(uint8_t idx);
};

} // namespace piratebot
