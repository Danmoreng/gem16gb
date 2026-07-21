#pragma once

#include <iostream>
#include <string_view>

namespace g4::test {

inline int failures = 0;

inline void Check(bool condition, std::string_view expression, std::string_view file, int line) {
  if (!condition) {
    std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

}  // namespace g4::test

#define G4_CHECK(expression) ::g4::test::Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

