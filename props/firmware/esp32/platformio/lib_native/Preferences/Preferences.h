#pragma once

#include <Arduino.h>
#include <cstring>
#include <map>
#include <string>

// Mock Preferences library for native unit tests. Stores data in memory only.

class Preferences {
public:
    bool begin(const char* name, bool readOnly = false) {
        (void)name;
        (void)readOnly;
        return true;
    }
    void end() {}
    bool clear() {
        store_.clear();
        return true;
    }

    size_t putString(const char* key, const char* value) {
        store_[key] = std::string(value);
        return store_[key].size();
    }
    size_t getString(const char* key, char* out, size_t maxLen) {
        auto it = store_.find(key);
        if (it == store_.end()) {
            if (maxLen > 0) out[0] = '\0';
            return 0;
        }
        size_t n = it->second.size();
        if (n >= maxLen) n = maxLen - 1;
        memcpy(out, it->second.c_str(), n);
        out[n] = '\0';
        return n;
    }

    size_t putUInt(const char* key, uint32_t value) {
        uint8_t buf[4] = {
            (uint8_t)(value >> 24),
            (uint8_t)(value >> 16),
            (uint8_t)(value >> 8),
            (uint8_t)(value)
        };
        store_[key] = std::string((char*)buf, 4);
        return 4;
    }
    uint32_t getUInt(const char* key, uint32_t defaultValue = 0) {
        auto it = store_.find(key);
        if (it == store_.end() || it->second.size() != 4) return defaultValue;
        const uint8_t* buf = (const uint8_t*)it->second.data();
        return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
               ((uint32_t)buf[2] << 8) | (uint32_t)buf[3];
    }

    size_t putUShort(const char* key, uint16_t value) {
        uint8_t buf[2] = {(uint8_t)(value >> 8), (uint8_t)(value)};
        store_[key] = std::string((char*)buf, 2);
        return 2;
    }
    uint16_t getUShort(const char* key, uint16_t defaultValue = 0) {
        auto it = store_.find(key);
        if (it == store_.end() || it->second.size() != 2) return defaultValue;
        const uint8_t* buf = (const uint8_t*)it->second.data();
        return ((uint16_t)buf[0] << 8) | (uint16_t)buf[1];
    }

    size_t putBool(const char* key, bool value) {
        store_[key] = std::string(1, value ? 1 : 0);
        return 1;
    }
    bool getBool(const char* key, bool defaultValue = false) {
        auto it = store_.find(key);
        if (it == store_.end() || it->second.size() != 1) return defaultValue;
        return it->second[0] != 0;
    }

private:
    std::map<std::string, std::string> store_;
};
