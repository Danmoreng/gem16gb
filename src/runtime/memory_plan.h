#pragma once

#include "gem16gb/memory.h"
#include "gem16gb/status.h"
#include "gem16gb/types.h"
#include "model/config.h"

namespace gem16gb::internal {

[[nodiscard]] Result<std::uint64_t> ContextTokens(ContextProfile profile);
[[nodiscard]] Result<MemoryPlan> BuildMemoryPlan(
    const ModelConfig& config,
    const ModelManifest& manifest,
    const MemoryPlanOptions& options);

}  // namespace gem16gb::internal
