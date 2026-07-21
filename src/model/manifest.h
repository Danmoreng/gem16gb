#pragma once

#include <filesystem>

#include "gem16gb/status.h"
#include "gem16gb/types.h"
#include "model/config.h"

namespace gem16gb::internal {

[[nodiscard]] Result<ModelManifest> BuildManifest(
    const std::filesystem::path& model_directory,
    const ModelConfig& config,
    bool validate);

}  // namespace gem16gb::internal

