#include "sensor_engine.h"

namespace piratebot {

SensorEngine::SensorEngine(SensorCallback cb) : callback_(cb) {
    for (int i = 0; i < MAX_PROFILES; ++i) {
        profiles_[i] = nullptr;
        last_state_[i] = 0xFF;
        last_change_ms_[i] = 0;
        last_emit_ms_[i] = 0;
    }
}

void SensorEngine::addProfile(const SensorProfile* profile) {
    if (profile_count_ >= MAX_PROFILES || !profile) return;
    pinMode(profile->pin, INPUT_PULLUP);
    profiles_[profile_count_++] = profile;
}

void SensorEngine::update() {
    uint32_t now = millis();
    for (uint8_t i = 0; i < profile_count_; ++i) {
        const SensorProfile* p = profiles_[i];
        bool raw = digitalRead(p->pin) == HIGH;
        bool detected = p->active_high ? raw : !raw;
        uint8_t state = detected ? 1 : 0;

        if (state != last_state_[i]) {
            if (now - last_change_ms_[i] >= p->debounce_ms) {
                last_state_[i] = state;
                last_change_ms_[i] = now;
                if (detected && now - last_emit_ms_[i] >= p->retrigger_ms) {
                    emit(p, true);
                    last_emit_ms_[i] = now;
                } else if (!detected) {
                    emit(p, false);
                }
            }
        }
    }
}

void SensorEngine::emit(const SensorProfile* profile, bool detected) {
    if (!callback_) return;
    Message msg;
    strlcpy(msg.topic, profile->topic, sizeof(msg.topic));
    strlcpy(msg.source, MESH_NODE_ID, sizeof(msg.source));
    msg.payload["detected"] = detected;
    msg.payload["pin"] = profile->pin;
    msg.meta.seq = ++seq_;
    msg.timestamp = millis() / 1000.0;
    callback_(msg);
}

} // namespace piratebot
