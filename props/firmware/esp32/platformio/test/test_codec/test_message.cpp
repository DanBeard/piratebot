#include <unity.h>
#include "codec/message.h"

// Include implementation directly so native tests link without requiring
// the full Arduino/ESP32 build environment.
#include "../../src/codec/message.cpp"

using namespace piratebot;

void setUp(void) {}
void tearDown(void) {}

void test_json_roundtrip(void) {
    Message msg;
    strlcpy(msg.topic, "effects.thunder.clap", sizeof(msg.topic));
    strlcpy(msg.source, "broker", sizeof(msg.source));
    strlcpy(msg.target, "thunder_01", sizeof(msg.target));
    msg.payload["duration_ms"] = 200;
    msg.payload["flash_count"] = 3;
    msg.timing.delay_ms = 150;
    msg.timing.has_delay = true;
    msg.meta.seq = 42;
    msg.timestamp = 1700000000.0;

    char buf[512];
    bool ok = msg.toJson(buf, sizeof(buf));
    TEST_ASSERT_TRUE(ok);

    Message out;
    bool parsed = out.fromJson(buf);
    TEST_ASSERT_TRUE(parsed);
    TEST_ASSERT_EQUAL_STRING("effects.thunder.clap", out.topic);
    TEST_ASSERT_EQUAL_STRING("broker", out.source);
    TEST_ASSERT_EQUAL_STRING("thunder_01", out.target);
    TEST_ASSERT_EQUAL(200, out.payload["duration_ms"]);
    TEST_ASSERT_EQUAL(3, out.payload["flash_count"]);
    TEST_ASSERT_TRUE(out.timing.has_delay);
    TEST_ASSERT_EQUAL(150, out.timing.delay_ms);
    TEST_ASSERT_EQUAL(42, out.meta.seq);
}

void test_framed_json_roundtrip(void) {
    Message msg;
    strlcpy(msg.topic, "effects.smoke.burst", sizeof(msg.topic));
    strlcpy(msg.source, "smoke_01", sizeof(msg.source));
    msg.payload["duration_ms"] = 1000;

    uint8_t buf[512];
    size_t len = 0;
    bool ok = msg.toFramed(buf, len, MESH_CODEC_JSON, MESH_FLAG_FLOOD_MESH);
    TEST_ASSERT_TRUE(ok);
    TEST_ASSERT_EQUAL(MESH_MAGIC_0, buf[0]);
    TEST_ASSERT_EQUAL(MESH_MAGIC_1, buf[1]);
    TEST_ASSERT_EQUAL(MESH_VERSION, buf[2]);
    TEST_ASSERT_EQUAL(MESH_FLAG_FLOOD_MESH, buf[3]);
    TEST_ASSERT_EQUAL(MESH_CODEC_JSON, buf[4]);

    Message out;
    bool parsed = out.fromFramed(buf, len);
    TEST_ASSERT_TRUE(parsed);
    TEST_ASSERT_EQUAL_STRING("effects.smoke.burst", out.topic);
    TEST_ASSERT_EQUAL_STRING("smoke_01", out.source);
    TEST_ASSERT_EQUAL(1000, out.payload["duration_ms"]);
}

void test_target_optional(void) {
    Message msg;
    strlcpy(msg.topic, "scene.idle", sizeof(msg.topic));
    strlcpy(msg.source, "broker", sizeof(msg.source));

    char buf[256];
    TEST_ASSERT_TRUE(msg.toJson(buf, sizeof(buf)));
    TEST_ASSERT_NULL(strstr(buf, "\"target\""));

    Message out;
    TEST_ASSERT_TRUE(out.fromJson(buf));
    TEST_ASSERT_EQUAL_STRING("", out.target);
}

void test_unknown_framed_version_fails(void) {
    uint8_t bad[] = {MESH_MAGIC_0, MESH_MAGIC_1, 0xFF, 0x00, MESH_CODEC_JSON, 0x00, 0x00};
    Message out;
    TEST_ASSERT_FALSE(out.fromFramed(bad, sizeof(bad)));
}

void test_bad_magic_fails(void) {
    uint8_t bad[] = {0x00, 0x00, MESH_VERSION, 0x00, MESH_CODEC_JSON, 0x00, 0x00};
    Message out;
    TEST_ASSERT_FALSE(out.fromFramed(bad, sizeof(bad)));
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_json_roundtrip);
    RUN_TEST(test_framed_json_roundtrip);
    RUN_TEST(test_target_optional);
    RUN_TEST(test_unknown_framed_version_fails);
    RUN_TEST(test_bad_magic_fails);
    return UNITY_END();
}
