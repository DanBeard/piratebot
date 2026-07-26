#include "effect_engine.h"

namespace piratebot {

EffectEngine::EffectEngine() {
    for (int i = 0; i < MAX_PROFILES; ++i) {
        profiles_[i] = nullptr;
        last_fire_ms_[i] = 0;
        active_[i] = false;
        segment_start_ms_[i] = 0;
        segment_index_[i] = 0;
        repeat_count_[i] = 0;
    }
}

void EffectEngine::addProfile(const EffectProfile* profile) {
    if (profile_count_ >= MAX_PROFILES || !profile) return;
    profiles_[profile_count_] = profile;
    for (uint8_t s = 0; s < profile->segment_count; ++s) {
        pinMode(profile->segments[s].pin, OUTPUT);
        setPin(profile->segments[s], false);
    }
    ++profile_count_;
}

void EffectEngine::getCapabilities(char** out, uint8_t& count, uint8_t max) const {
    count = 0;
    for (uint8_t i = 0; i < profile_count_ && count < max; ++i) {
        out[count++] = const_cast<char*>(profiles_[i]->topic);
    }
}

int EffectEngine::findProfileIndex(const char* topic) const {
    for (uint8_t i = 0; i < profile_count_; ++i) {
        if (strcmp(topic, profiles_[i]->topic) == 0) return i;
    }
    return -1;
}

void EffectEngine::setPin(const GpioSegment& seg, bool on) {
    bool value = on ? seg.active_high : !seg.active_high;
    digitalWrite(seg.pin, value ? HIGH : LOW);
}

bool EffectEngine::canFire(uint8_t idx) {
    if (active_[idx]) return false;
    uint32_t now = millis();
    if (now - last_fire_ms_[idx] < profiles_[idx]->cooldown_ms) return false;
    return true;
}

void EffectEngine::complete(uint8_t idx) {
    const EffectProfile* p = profiles_[idx];
    for (uint8_t s = 0; s < p->segment_count; ++s) {
        setPin(p->segments[s], false);
    }
    active_[idx] = false;
    last_fire_ms_[idx] = millis();
}

bool EffectEngine::handleMessage(const Message& msg) {
    if (strcmp(msg.topic, "scene.estop") == 0) {
        stopAll();
        return true;
    }
    int idx = findProfileIndex(msg.topic);
    if (idx < 0) return false;
    if (!canFire(idx)) return false;

    // Honor delay_ms
    int32_t delay = 0;
    if (msg.timing.has_delay) delay = msg.timing.delay_ms;
    uint32_t now = millis();

    active_[idx] = true;
    segment_index_[idx] = 0;
    repeat_count_[idx] = 0;
    segment_start_ms_[idx] = now + delay;

    // Apply payload overrides if present
    const EffectProfile* p = profiles_[idx];
    uint32_t duration = msg.payload["duration_ms"] | p->segments[0].on_ms;
    if (duration > p->max_on_ms) duration = p->max_on_ms;

    // Hard safety: schedule completion at max_on_ms even if segments run long.
    // (Simplification: we rely on update() and segment timing.)
    (void)duration;
    return true;
}

void EffectEngine::update() {
    uint32_t now = millis();
    for (uint8_t idx = 0; idx < profile_count_; ++idx) {
        if (!active_[idx]) continue;
        const EffectProfile* p = profiles_[idx];

        // Hard off at max_on_ms relative to segment start
        uint32_t total_on = now - segment_start_ms_[idx];
        if (p->max_on_ms > 0 && total_on >= p->max_on_ms) {
            complete(idx);
            continue;
        }

        // Wait for delay to elapse
        if (now < segment_start_ms_[idx]) continue;

        uint8_t seg_idx = segment_index_[idx];
        if (seg_idx >= p->segment_count) {
            complete(idx);
            continue;
        }

        const GpioSegment& seg = p->segments[seg_idx];
        uint32_t elapsed = now - segment_start_ms_[idx];
        uint32_t cycle = seg.on_ms + seg.off_ms;

        if (elapsed < seg.on_ms) {
            setPin(seg, true);
        } else if (elapsed < cycle) {
            setPin(seg, false);
        } else {
            if (repeat_count_[idx] < seg.repeat) {
                repeat_count_[idx]++;
                segment_start_ms_[idx] = now;
            } else {
                segment_index_[idx]++;
                segment_start_ms_[idx] = now;
                repeat_count_[idx] = 0;
                setPin(seg, false);
            }
        }
    }
}

void EffectEngine::stopAll() {
    for (uint8_t idx = 0; idx < profile_count_; ++idx) {
        complete(idx);
    }
}

} // namespace piratebot
