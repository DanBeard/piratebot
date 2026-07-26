#include "discovery.h"

#include <ArduinoJson.hpp>
#include <lwip/sockets.h>
#include <lwip/inet.h>
#include <lwip/ip_addr.h>

namespace piratebot {

void DiscoveryClient::begin(uint16_t port) {
    port_ = port;
}

bool DiscoveryClient::beginSocket() {
    sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock_ < 0) return false;

    int yes = 1;
    setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock_);
        sock_ = -1;
        return false;
    }

    ip4_addr_t mcast;
    IP4_ADDR(&mcast, 239, 255, 42, 99);
    if (igmp_joingroup(IP_ADDR_ANY, &mcast) != ERR_OK) {
        close(sock_);
        sock_ = -1;
        return false;
    }

    int flags = fcntl(sock_, F_GETFL, 0);
    fcntl(sock_, F_SETFL, flags | O_NONBLOCK);
    return true;
}

bool DiscoveryClient::findBroker(char* out_host, size_t host_len, uint16_t& out_port, uint32_t timeout_ms) {
    if (!beginSocket()) return false;

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port_);
    inet_aton("239.255.42.99", &dest.sin_addr);

    char request[128];
    snprintf(request, sizeof(request), "{\"cmd\":\"discover\",\"session\":\"%s\"}", MESH_SESSION);

    uint32_t start = millis();
    while (millis() - start < timeout_ms) {
        sendto(sock_, request, strlen(request), 0, (struct sockaddr*)&dest, sizeof(dest));

        char reply[256];
        struct sockaddr_in src;
        socklen_t src_len = sizeof(src);
        int n = recvfrom(sock_, reply, sizeof(reply) - 1, 0, (struct sockaddr*)&src, &src_len);
        if (n > 0) {
            reply[n] = '\0';
            JsonDocument doc;
            if (deserializeJson(doc, reply) == DeserializationError::Ok) {
                const char* broker = doc["broker"];
                if (broker && strncmp(broker, "ws://", 5) == 0) {
                    // broker is ws://host:port/ws
                    const char* host_start = broker + 5;
                    const char* colon = strchr(host_start, ':');
                    if (colon) {
                        size_t host_len_actual = colon - host_start;
                        if (host_len_actual >= host_len) host_len_actual = host_len - 1;
                        strncpy(out_host, host_start, host_len_actual);
                        out_host[host_len_actual] = '\0';
                        out_port = atoi(colon + 1);
                        close(sock_);
                        sock_ = -1;
                        return true;
                    }
                }
            }
        }
        delay(500);
    }

    close(sock_);
    sock_ = -1;
    return false;
}

} // namespace piratebot
