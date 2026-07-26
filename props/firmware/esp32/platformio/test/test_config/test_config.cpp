#include <unity.h>
#include "config/config_manager.h"
#include "config/config_handler.h"
#include "ota/ota_manager.h"

#include "../../src/codec/message.cpp"
#include "../../src/config/config_manager.cpp"
#include "../../src/config/config_handler.cpp"
#include "../../src/ota/ota_manager.cpp"

using namespace piratebot;

void setUp(void) {
    ConfigManager mgr;
    mgr.begin();
    mgr.clear();
}
void tearDown(void) {}

void test_config_load_defaults(void) {
    ConfigManager mgr;
    PropConfig cfg;
    // With a clean store, defaults come from compile-time defines.
    mgr.load(cfg);
    TEST_ASSERT_EQUAL_STRING(MESH_NODE_ID, cfg.node_id);
    TEST_ASSERT_EQUAL_STRING(WIFI_SSID, cfg.wifi_ssid);
    TEST_ASSERT_EQUAL(BROKER_WS_PORT, cfg.broker_ws_port);
    TEST_ASSERT_FALSE(cfg.ota_enabled);
}

void test_config_save_and_load(void) {
    ConfigManager mgr;
    PropConfig cfg;
    strlcpy(cfg.profile, "smoke", sizeof(cfg.profile));
    strlcpy(cfg.broker_host, "192.168.0.99", sizeof(cfg.broker_host));
    cfg.broker_ws_port = 19001;
    cfg.ota_enabled = true;
    TEST_ASSERT_TRUE(mgr.save(cfg));

    PropConfig loaded;
    mgr.load(loaded);
    TEST_ASSERT_EQUAL_STRING("smoke", loaded.profile);
    TEST_ASSERT_EQUAL_STRING("192.168.0.99", loaded.broker_host);
    TEST_ASSERT_EQUAL(19001, loaded.broker_ws_port);
    TEST_ASSERT_TRUE(loaded.ota_enabled);
}

void test_config_get_handler(void) {
    ConfigManager mgr;
    PropConfig cfg;
    strlcpy(cfg.profile, "cannon", sizeof(cfg.profile));
    mgr.save(cfg);

    Message req, reply;
    strlcpy(req.topic, "prop.config.get", sizeof(req.topic));
    strlcpy(req.source, "control_center", sizeof(req.source));
    strlcpy(req.target, "test_node", sizeof(req.target));

    ConfigHandler handler;
    bool handled = handler.handle(req, reply);
    TEST_ASSERT_TRUE(handled);
    TEST_ASSERT_EQUAL_STRING("prop.config.current", reply.topic);
    TEST_ASSERT_EQUAL_STRING("cannon", reply.payload["profile"]);
    // Secrets must not leak
    TEST_ASSERT_FALSE(reply.payload.containsKey("wifi_pass"));
}

void test_config_set_handler(void) {
    Message req, reply;
    strlcpy(req.topic, "prop.config.set", sizeof(req.topic));
    strlcpy(req.source, "control_center", sizeof(req.source));
    req.payload["profile"] = "strobe";
    req.payload["broker_host"] = "192.168.0.77";
    req.payload["ota_enabled"] = false;

    ConfigHandler handler;
    bool handled = handler.handle(req, reply);
    TEST_ASSERT_TRUE(handled);
    TEST_ASSERT_TRUE(reply.payload["success"]);
    TEST_ASSERT_TRUE(handler.needsReboot());

    ConfigManager mgr;
    PropConfig loaded;
    mgr.load(loaded);
    TEST_ASSERT_EQUAL_STRING("strobe", loaded.profile);
    TEST_ASSERT_EQUAL_STRING("192.168.0.77", loaded.broker_host);
    TEST_ASSERT_FALSE(loaded.ota_enabled);
}

void test_ota_disabled_by_default(void) {
    OtaManager ota;
    TEST_ASSERT_FALSE(ota.enabled());
}

void test_ota_enable_requires_confirm(void) {
    OtaManager ota;
    Message req, reply;
    strlcpy(req.topic, "prop.ota.enable", sizeof(req.topic));
    strlcpy(req.source, "control_center", sizeof(req.source));
    req.payload["enabled"] = true;
    // confirm missing

    bool handled = ota.handle(req, reply, "idle");
    TEST_ASSERT_TRUE(handled);
    TEST_ASSERT_FALSE(ota.enabled());
    TEST_ASSERT_TRUE(reply.payload.containsKey("error"));
}

void test_ota_start_blocked_when_disabled(void) {
    OtaManager ota;
    Message req, reply;
    strlcpy(req.topic, "prop.ota.start", sizeof(req.topic));
    strlcpy(req.source, "control_center", sizeof(req.source));
    req.payload["url"] = "http://192.168.0.2/firmware.bin";

    bool handled = ota.handle(req, reply, "idle");
    TEST_ASSERT_TRUE(handled);
    TEST_ASSERT_TRUE(reply.payload.containsKey("error"));
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_config_load_defaults);
    RUN_TEST(test_config_save_and_load);
    RUN_TEST(test_config_get_handler);
    RUN_TEST(test_config_set_handler);
    RUN_TEST(test_ota_disabled_by_default);
    RUN_TEST(test_ota_enable_requires_confirm);
    RUN_TEST(test_ota_start_blocked_when_disabled);
    return UNITY_END();
}
