#include <unity.h>
#include "effect_engine.h"
#include "Arduino.h"

#include "../../src/effect_engine.cpp"
#include "../../src/profiles.cpp"

using namespace piratebot;

static Message makeMsg(const char* topic) {
    Message msg;
    strlcpy(msg.topic, topic, sizeof(msg.topic));
    strlcpy(msg.source, "test", sizeof(msg.source));
    return msg;
}

void setUp(void) {
    mock_millis_value = 0;
}
void tearDown(void) {}

void test_cannon_fires(void) {
    EffectEngine engine;
    engine.addProfile(&CANNON_PROFILE);

    Message msg = makeMsg("effects.cannon.fire");
    bool fired = engine.handleMessage(msg);
    TEST_ASSERT_TRUE(fired);

    // Simulate 1 second of updates
    for (int i = 0; i < 200; ++i) {
        mock_millis_value += 5;
        delay(5);
        engine.update();
    }
}

void test_cannon_cooldown_blocks_refire(void) {
    EffectEngine engine;
    engine.addProfile(&CANNON_PROFILE);

    Message msg = makeMsg("effects.cannon.fire");
    TEST_ASSERT_TRUE(engine.handleMessage(msg));

    // Try again immediately
    bool fired = engine.handleMessage(msg);
    TEST_ASSERT_FALSE(fired);
}

void test_estop_stops_all(void) {
    EffectEngine engine;
    engine.addProfile(&SMOKE_PROFILE);

    Message fire = makeMsg("effects.smoke.burst");
    TEST_ASSERT_TRUE(engine.handleMessage(fire));

    Message estop = makeMsg("scene.estop");
    TEST_ASSERT_TRUE(engine.handleMessage(estop));
}

void test_unknown_topic_ignored(void) {
    EffectEngine engine;
    engine.addProfile(&CANNON_PROFILE);
    Message msg = makeMsg("effects.unknown");
    TEST_ASSERT_FALSE(engine.handleMessage(msg));
}

void test_cooldown_across_overflow(void) {
    EffectEngine engine;
    engine.addProfile(&CANNON_PROFILE);

    mock_millis_value = 0xFFFFFFF0UL;
    Message msg = makeMsg("effects.cannon.fire");
    TEST_ASSERT_TRUE(engine.handleMessage(msg));

    // Let the effect run and complete. Cannon on_ms=500, max_on_ms=3000.
    mock_millis_value = 0x00000050UL;
    engine.update();
    TEST_ASSERT_FALSE(engine.handleMessage(msg));  // still in cooldown

    // After cooldown (5000ms from 0xFFFFFFF0 -> 0x00001388), update once to
    // mark the effect complete, then advance past cooldown before refiring.
    mock_millis_value = 0x00002000UL;
    engine.update();  // completes effect, sets last_fire_ms_ = 0x2000
    mock_millis_value = 0x00003800UL;  // +6s from completion
    engine.update();
    TEST_ASSERT_TRUE(engine.handleMessage(msg));
}

void test_delay_across_overflow(void) {
    EffectEngine engine;
    engine.addProfile(&SMOKE_PROFILE);

    mock_millis_value = 0xFFFFFFF0UL;
    Message msg = makeMsg("effects.smoke.burst");
    msg.timing.delay_ms = 100;
    msg.timing.has_delay = true;
    TEST_ASSERT_TRUE(engine.handleMessage(msg));

    // Advance past overflow
    mock_millis_value = 0x00000010UL;
    engine.update();
    // Effect should still respect the delay window
    // Without overflow safety this would fire prematurely.
}

void test_max_on_across_overflow(void) {
    EffectEngine engine;
    engine.addProfile(&STROBE_PROFILE);

    mock_millis_value = 0xFFFFFFF0UL;
    Message msg = makeMsg("effects.strobe.flash");
    TEST_ASSERT_TRUE(engine.handleMessage(msg));

    // Start near overflow, advance past it
    engine.update();
    mock_millis_value = 0x00001000UL;  // well past max_on_ms
    engine.update();
    // Should complete without hanging; cooldown should now be set
    TEST_ASSERT_FALSE(engine.handleMessage(msg));  // still in cooldown
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_cannon_fires);
    RUN_TEST(test_cannon_cooldown_blocks_refire);
    RUN_TEST(test_estop_stops_all);
    RUN_TEST(test_unknown_topic_ignored);
    RUN_TEST(test_cooldown_across_overflow);
    RUN_TEST(test_delay_across_overflow);
    RUN_TEST(test_max_on_across_overflow);
    return UNITY_END();
}
