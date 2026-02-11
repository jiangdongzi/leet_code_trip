#include <cstdint>
#include <iostream>
#include <random>
#include <string>

#include "dec_str.hpp"

static int failures = 0;

static void expectEq(const std::string& a, const std::string& b, const std::string& expected) {
    const std::string got = B::decStr(a, b);
    if (got != expected) {
        failures++;
        std::cerr << "FAIL decStr(\"" << a << "\", \"" << b << "\") => \"" << got << "\"; expected \""
                  << expected << "\"\n";
    }
}

static std::string toDecStr(long long x) {
    return std::to_string(x);
}

int main() {
    // Deterministic edge cases
    expectEq("0", "0", "0");
    expectEq("1", "0", "1");
    expectEq("0", "1", "-1");
    expectEq("1", "1", "0");
    expectEq("10", "1", "9");
    expectEq("1000", "1", "999");
    expectEq("1000", "999", "1");
    expectEq("999", "1000", "-1");
    expectEq("100", "99", "1");
    expectEq("99", "100", "-1");
    expectEq("12345678901234567890", "12345678901234567889", "1");
    expectEq("500", "500", "0");
    expectEq("10000000000000000000", "1", "9999999999999999999");

    // Leading-zero normalization (inputs are still non-negative)
    expectEq("0000", "0", "0");
    expectEq("0", "0000", "0");
    expectEq("0001", "1", "0");
    expectEq("0010", "0009", "1");
    expectEq("0000", "0001", "-1");

    // Random small-number fuzzing against signed arithmetic
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<long long> dist(0, 1000000000LL);
    for (int i = 0; i < 2000; i++) {
        const long long x = dist(rng);
        const long long y = dist(rng);
        const std::string a = std::to_string(x);
        const std::string b = std::to_string(y);
        const std::string expected = toDecStr(x - y);
        const std::string got = B::decStr(a, b);
        if (got != expected) {
            failures++;
            std::cerr << "FAIL fuzz decStr(\"" << a << "\", \"" << b << "\") => \"" << got
                      << "\"; expected \"" << expected << "\"\n";
            break;
        }
    }

    if (failures == 0) {
        std::cout << "OKK\n";
        return 0;
    }
    return 1;
}

