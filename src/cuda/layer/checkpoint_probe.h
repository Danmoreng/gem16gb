#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "gem16gb/status.h"

namespace gem16gb::internal {

struct LocalAttentionProbeSample {
  std::uint64_t element = 0;
  float cuda_reference = 0.0F;
  float sm120_direct = 0.0F;
};

struct LocalAttentionCheckpointProbeResult {
  std::uint64_t context_tokens = 0;
  std::uint64_t device_bytes = 0;
  double reference_native_max_abs = 0.0;
  double reference_native_rms = 0.0;
  double reference_native_cosine = 0.0;
  std::vector<LocalAttentionProbeSample> samples;
};

[[nodiscard]] Result<LocalAttentionCheckpointProbeResult>
RunLayer0LocalAttentionCheckpointProbe(const std::filesystem::path& model_directory);

}  // namespace gem16gb::internal
