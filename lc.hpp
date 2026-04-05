#pragma once

#include <iostream>
#include <string>

template <typename... Args> inline void fp(const std::string &format, Args &&...args) {
    (void)sizeof...(args);
    std::cout << format;
}
