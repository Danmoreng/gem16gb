#pragma once

#include <cstdint>

namespace g4 {

struct SamplingOptions {
  float temperature = 1.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  float repetition_penalty = 1.0F;
  std::uint64_t seed = 0;
};

}  // namespace g4
