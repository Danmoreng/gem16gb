#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "gem16gb/status.h"

namespace gem16gb::internal {

[[nodiscard]] Status LaunchRmsNorm(const float* input,
                                   const std::uint16_t* weight_bf16,
                                   float* output,
                                   std::uint64_t vectors,
                                   std::uint64_t width,
                                   float epsilon,
                                   cudaStream_t stream);

[[nodiscard]] Status LaunchRotaryEmbedding(float* states,
                                           std::uint64_t heads,
                                           std::uint64_t head_dimension,
                                           std::uint64_t rotary_dimensions,
                                           std::uint64_t position,
                                           double theta,
                                           cudaStream_t stream);

[[nodiscard]] Status LaunchProportionalRotaryEmbedding(float* states,
                                                       std::uint64_t heads,
                                                       std::uint64_t head_dimension,
                                                       double rotary_factor,
                                                       std::uint64_t position,
                                                       double theta,
                                                       double scaling_factor,
                                                       cudaStream_t stream);

[[nodiscard]] Status LaunchAppendKv(const float* key,
                                    const float* value,
                                    float* key_cache,
                                    float* value_cache,
                                    std::uint64_t slot,
                                    std::uint64_t kv_heads,
                                    std::uint64_t head_dimension,
                                    cudaStream_t stream);

// Correctness-first batch-one decode attention. `scores` is a caller-owned
// workspace of query_heads * tokens floats and is overwritten with probabilities.
// Computes grouped-query attention over the exact cache view supplied by the
// caller. Sliding versus full attention is determined by that view's extent.
[[nodiscard]] Status LaunchLocalAttentionDecode(const float* query,
                                                const float* key_cache,
                                                const float* value_cache,
                                                float* scores,
                                                float* output,
                                                std::uint64_t query_heads,
                                                std::uint64_t kv_heads,
                                                std::uint64_t head_dimension,
                                                std::uint64_t tokens,
                                                cudaStream_t stream);

[[nodiscard]] Status LaunchScale(float* values,
                                 const std::uint16_t* scalar_bf16,
                                 std::uint64_t elements,
                                 cudaStream_t stream);

}  // namespace gem16gb::internal
