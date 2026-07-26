#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <string>

// Mock Arduino types and functions for native unit tests.

using String = std::string;

#define HIGH 1
#define LOW 0
#define INPUT_PULLUP 0x02
#define OUTPUT 0x01
#define INPUT 0x00

extern unsigned long mock_millis_value;
inline unsigned long millis() {
    return mock_millis_value;
}
inline void setMockMillis(unsigned long v) { mock_millis_value = v; }

inline void delay(unsigned long) {}

inline void pinMode(uint8_t, uint8_t) {}
inline void digitalWrite(uint8_t, uint8_t) {}
inline int digitalRead(uint8_t) { return LOW; }

inline void Serial_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

struct MockSerial {
    template<typename... Args>
    void printf(const char* fmt, Args... args) {
        Serial_printf(fmt, args...);
    }
    void println(const char* msg) {
        puts(msg);
    }
    void println() {
        puts("");
    }
};

extern MockSerial Serial;
