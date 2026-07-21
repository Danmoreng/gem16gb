#pragma once

#include <filesystem>
#include <iosfwd>

#include "g4/status.h"
#include "g4/types.h"

namespace g4 {

struct InspectOptions {
  std::filesystem::path model_directory;
  bool validate = false;
};

[[nodiscard]] Result<ModelManifest> InspectCheckpoint(const InspectOptions& options);
[[nodiscard]] Status WriteManifestJson(const ModelManifest& manifest, std::ostream& output);
void PrintManifestSummary(const ModelManifest& manifest, std::ostream& output);

}  // namespace g4

