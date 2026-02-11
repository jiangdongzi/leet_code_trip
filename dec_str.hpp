#pragma once

#include <algorithm>
#include <string>

namespace B {

inline std::string decStr(std::string a, std::string b) {
    auto isLess = [](const std::string& a, const std::string& b) -> bool {
        if (a.size() == b.size()) return a < b;
        return a.size() < b.size();
    };
    std::string ret;
    auto dec = [](const std::string& subtrahend, const std::string& minuend) -> std::string {
        const int sz = subtrahend.size(), mz = minuend.size();
        int borrow = 0;
        int idx = 0;
        std::string ret;
        while (idx < sz) {
            const int sVal = subtrahend[sz - 1 - idx] - '0';
            const int mVal = idx < mz ? minuend[mz - 1 - idx] - '0' : 0;
            const int x = (sVal - mVal - borrow + 10) % 10;
            ret.push_back(x + '0');
            borrow = sVal - mVal - borrow < 0 ? 1 : 0;
            idx++;
        }
        while (!ret.empty() && ret.back() == '0') ret.pop_back();
        std::reverse(ret.begin(), ret.end());
        if (ret.empty()) ret.push_back('0');
        return ret;
    };
    if (isLess(a, b)) {
        ret = dec(b, a);
        if (ret[0] != '0') {
            ret.insert(0, "-");
        }
    } else {
        ret = dec(a, b);
    }
    return ret;
}

} // namespace B

