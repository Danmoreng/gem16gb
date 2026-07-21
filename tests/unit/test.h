#pragma once

#include <iostream>
#include <string_view>

namespace gem16gb::test {

inline int failures = 0;

inline void Check(bool condition, std::string_view expression, std::string_view file, int line) {
  if (!condition) {
    std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

}  // namespace gem16gb::test

#define GEM16GB_CHECK(expression) ::gem16gb::test::Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

