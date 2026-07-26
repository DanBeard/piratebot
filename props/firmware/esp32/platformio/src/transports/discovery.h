#pragma once

#include <Arduino.h>

namespace piratebot {

// UDP multicast discovery: shout AHOY and listen for broker reply.
class DiscoveryClient {
public:
    void begin(uint16_t port);
    bool findBroker(char* out_host, size_t host_len, uint16_t& out_port, uint32_t timeout_ms = 5000);

private:
    int sock_ = -1;
    uint16_t port_ = 9002;
    bool beginSocket();
};

} // namespace piratebot
