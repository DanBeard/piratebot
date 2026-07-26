#include <unity.h>
#include "effect_engine.h"

#include "../../src/effect_engine.cpp"
#include "../../src/profiles.cpp"

using namespace piratebot;

static Message makeMsg(const char* topic) {
    Message msg;
    strlcpy(msg.topic, topic, sizeof(msg.topic));
    strlcpy(msg.source, "test", sizeof(msg.source));
    return msg;
}

void setUp(void) {}
void tearDown(void) {}

void test_cannon_fires(void) {
    EffectEngine engine;
    engine.addProfile(&CANNON_PROFILE);

    Message msg = makeMsg("effects.cannon.fire");
    bool fired = engine.handleMessage(msg);
    TEST_ASSERT_TRUE(fired);

    // Simulate 1 second of updates
    for (int i = 0; i < 200; ++i) {
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

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_cannon_fires);
    RUN_TEST(test_cannon_cooldown_blocks_refire);
    RUN_TEST(test_estop_stops_all);
    RUN_TEST(test_unknown_topic_ignored);
    return UNITY_END();
}
