#include "profiles.h"

namespace piratebot {

const EffectProfile CANNON_PROFILE = {
    .id = "cannon",
    .name = "Cannon",
    .topic = "effects.cannon.fire",
    .segments = {
        {.pin = 12, .active_high = true, .on_ms = 500, .off_ms = 0, .repeat = 0},
    },
    .segment_count = 1,
    .max_on_ms = 3000,
    .cooldown_ms = 5000,
    .requires_ack = true,
};

const EffectProfile SMOKE_PROFILE = {
    .id = "smoke",
    .name = "Smoke Machine",
    .topic = "effects.smoke.burst",
    .segments = {
        {.pin = 13, .active_high = true, .on_ms = 2000, .off_ms = 0, .repeat = 0},
    },
    .segment_count = 1,
    .max_on_ms = 5000,
    .cooldown_ms = 3000,
    .requires_ack = true,
};

const EffectProfile STROBE_PROFILE = {
    .id = "strobe",
    .name = "Strobe",
    .topic = "effects.strobe.flash",
    .segments = {
        {.pin = 14, .active_high = true, .on_ms = 50, .off_ms = 50, .repeat = 9},
    },
    .segment_count = 1,
    .max_on_ms = 3000,
    .cooldown_ms = 2000,
    .requires_ack = false,
};

const EffectProfile THUNDER_PROFILE = {
    .id = "thunder",
    .name = "Thunder Clap",
    .topic = "effects.thunder.clap",
    .segments = {
        {.pin = 15, .active_high = true, .on_ms = 200, .off_ms = 0, .repeat = 0},
    },
    .segment_count = 1,
    .max_on_ms = 1000,
    .cooldown_ms = 1500,
    .requires_ack = false,
};

const EffectProfile RELAY_PROFILE = {
    .id = "relay",
    .name = "Generic Relay",
    .topic = "effects.relay.toggle",
    .segments = {
        {.pin = 16, .active_high = true, .on_ms = 1000, .off_ms = 0, .repeat = 0},
    },
    .segment_count = 1,
    .max_on_ms = 10000,
    .cooldown_ms = 1000,
    .requires_ack = false,
};

const SensorProfile PIR_PROFILE = {
    .id = "pir",
    .name = "PIR Motion",
    .topic = "sensors.pir.motion",
    .pin = 25,
    .active_high = true,
    .debounce_ms = 500,
    .retrigger_ms = 3000,
};

const SensorProfile BEAM_PROFILE = {
    .id = "beam",
    .name = "Beam Break",
    .topic = "sensors.beam.break",
    .pin = 26,
    .active_high = false,
    .debounce_ms = 50,
    .retrigger_ms = 2000,
};

static const EffectProfile* EFFECTS[] = {
    &CANNON_PROFILE, &SMOKE_PROFILE, &STROBE_PROFILE, &THUNDER_PROFILE, &RELAY_PROFILE,
};

static const SensorProfile* SENSORS[] = {
    &PIR_PROFILE, &BEAM_PROFILE,
};

const EffectProfile* findEffectProfile(const char* topic) {
    for (auto p : EFFECTS) {
        if (strcmp(topic, p->topic) == 0 || strcmp(topic, p->id) == 0) return p;
    }
    return nullptr;
}

const SensorProfile* findSensorProfile(const char* topic) {
    for (auto p : SENSORS) {
        if (strcmp(topic, p->topic) == 0 || strcmp(topic, p->id) == 0) return p;
    }
    return nullptr;
}

} // namespace piratebot
