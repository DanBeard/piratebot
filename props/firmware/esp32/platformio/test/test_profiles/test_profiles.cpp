#include <unity.h>
#include "profiles.h"

#include "../../src/profiles.cpp"

using namespace piratebot;

void setUp(void) {}
void tearDown(void) {}

void test_find_cannon_profile(void) {
    const EffectProfile* p = findEffectProfile("effects.cannon.fire");
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL_STRING("cannon", p->id);
    TEST_ASSERT_EQUAL_STRING("effects.cannon.fire", p->topic);
    TEST_ASSERT_EQUAL(500, p->segments[0].on_ms);
    TEST_ASSERT_EQUAL(5000, p->cooldown_ms);
}

void test_find_smoke_profile(void) {
    const EffectProfile* p = findEffectProfile("smoke");
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL_STRING("effects.smoke.burst", p->topic);
}

void test_find_strobe_profile(void) {
    const EffectProfile* p = findEffectProfile("effects.strobe.flash");
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL(10, p->segments[0].repeat + 1);  // repeat count is N additional cycles
}

void test_find_missing_profile(void) {
    const EffectProfile* p = findEffectProfile("effects.unknown");
    TEST_ASSERT_NULL(p);
}

void test_find_pir_profile(void) {
    const SensorProfile* p = findSensorProfile("sensors.pir.motion");
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL(25, p->pin);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_find_cannon_profile);
    RUN_TEST(test_find_smoke_profile);
    RUN_TEST(test_find_strobe_profile);
    RUN_TEST(test_find_missing_profile);
    RUN_TEST(test_find_pir_profile);
    return UNITY_END();
}
