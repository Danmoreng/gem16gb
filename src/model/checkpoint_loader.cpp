#include "g4/model.h"

#include <array>

#include "model/config.h"
#include "model/manifest.h"

namespace g4 {

Result<ModelManifest> InspectCheckpoint(const InspectOptions& options) {
  if (options.model_directory.empty()) {
    return Status(StatusCode::kInvalidArgument, "--model requires a checkpoint directory");
  }
  std::error_code error;
  if (!std::filesystem::is_directory(options.model_directory, error)) {
    return Status(StatusCode::kNotFound, "model directory does not exist: " + options.model_directory.string());
  }

  constexpr std::array required_files = {
      "config.json", "tokenizer.json", "tokenizer_config.json"};
  for (const char* name : required_files) {
    const auto path = options.model_directory / name;
    if (!std::filesystem::is_regular_file(path)) {
      return Status(StatusCode::kNotFound, "required checkpoint file is missing: " + path.string());
    }
  }
  auto config = internal::LoadModelConfig(options.model_directory / "config.json");
  if (!config.ok()) return config.status();
  if (options.validate) {
    auto validation = internal::ValidatePrimaryModelContract(config.value());
    if (!validation.ok()) return validation;
  }
  return internal::BuildManifest(options.model_directory, config.value(), options.validate);
}

}  // namespace g4

