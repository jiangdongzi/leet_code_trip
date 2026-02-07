#pragma once

#include <utility>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

template <typename... Args> inline void fp(fmt::format_string<Args...> format, Args &&...args) {
    fmt::print(format, std::forward<Args>(args)...);
}
