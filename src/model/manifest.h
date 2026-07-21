#pragma once

#include <filesystem>

#include "g4/status.h"
#include "g4/types.h"
#include "model/config.h"

namespace g4::internal {

[[nodiscard]] Result<ModelManifest> BuildManifest(
    const std::filesystem::path& model_directory,
    const ModelConfig& config,
    bool validate);

}  // namespace g4::internal

