#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <vector>

#include "gem16gb/status.h"

namespace gem16gb {

void PrintKernelCapabilities(std::ostream& output);

struct GreedyInferenceOptions {
  std::filesystem::path model_directory;
  std::vector<std::uint32_t> input_token_ids;
  std::vector<std::uint32_t> stop_token_ids;
  std::vector<std::uint32_t> suppressed_token_ids;
  std::filesystem::path logits_dump_path;
  std::uint64_t max_generated_tokens = 1;
  std::uint64_t max_context_tokens = 128;
};

struct GreedyInferenceResult {
  std::vector<std::uint32_t> output_token_ids;
  std::uint32_t stop_token_id = 0;
  double model_load_milliseconds = 0.0;
  double prompt_milliseconds = 0.0;
  double decode_milliseconds = 0.0;
  double decode_tokens_per_second = 0.0;
  std::uint64_t weight_arena_bytes = 0;
  std::uint64_t kv_cache_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t fallback_count = 0;
  std::uint64_t logits_dump_steps = 0;
  bool source_layout_direct = false;
  bool token_loop_allocations = false;
  bool benchmark_qualified = false;
  bool stopped = false;
  bool logits_dumped = false;
};

// Correctness-first, batch-one CUDA characterization. It accepts already-tokenized input,
// executes every decoder layer, and performs greedy selection on the GPU. The result remains
// explicitly unqualified until prompt-derived hidden-state and full-logit gates pass.
[[nodiscard]] Result<GreedyInferenceResult> RunGreedyInference(
    const GreedyInferenceOptions& options);
[[nodiscard]] Status WriteGreedyInferenceJson(const GreedyInferenceResult& result,
                                              std::ostream& output);

}  // namespace gem16gb
