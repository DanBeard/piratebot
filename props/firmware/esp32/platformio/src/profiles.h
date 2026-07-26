#pragma once

#include <Arduino.h>
#include "config.h"

namespace piratebot {

// Single-segment actuator effect with optional blink pattern.
struct GpioSegment {
    uint8_t pin;
    bool active_high;
    uint32_t on_ms;
    uint32_t off_ms;
    uint8_t repeat;
};

struct EffectProfile {
    const char *id;
    const char *name;
    const char *topic;
    GpioSegment segments[4];
    uint8_t segment_count;
    uint32_t max_on_ms;
    uint32_t cooldown_ms;
    bool requires_ack;
};

struct SensorProfile {
    const char *id;
    const char *name;
    const char *topic;
    uint8_t pin;
    bool active_high;
    uint32_t debounce_ms;
    uint32_t retrigger_ms;
};

// Pre-defined profiles. Select via NVS profile= name or compile flag.
extern const EffectProfile CANNON_PROFILE;
extern const EffectProfile SMOKE_PROFILE;
extern const EffectProfile STROBE_PROFILE;
extern const EffectProfile THUNDER_PROFILE;
extern const EffectProfile RELAY_PROFILE;
extern const SensorProfile PIR_PROFILE;
extern const SensorProfile BEAM_PROFILE;

const EffectProfile *findEffectProfile(const char *topic);
const SensorProfile *findSensorProfile(const char *topic);

} // namespace piratebot
