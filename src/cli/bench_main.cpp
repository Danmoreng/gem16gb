#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>

#include "gem16gb/memory.h"

namespace {

void Usage(std::ostream& output) {
  output << "Usage: gem16gb-bench <model-load|prefill|decode|end-to-end|kernel|memory|quality|mtp> [options]\n"
         << "\n"
         << "Memory mode:\n"
         << "  gem16gb-bench memory --model <checkpoint-dir>\n"
         << "      --profile <interactive|standard|long|xlong|max>\n"
         << "      --kv-storage <shared|separate> [--kv-element-bytes <1|2>]\n";
}

bool ParseProfile(std::string_view value, gem16gb::ContextProfile& profile) {
  if (value == "interactive") profile = gem16gb::ContextProfile::kInteractive;
  else if (value == "standard") profile = gem16gb::ContextProfile::kStandard;
  else if (value == "long") profile = gem16gb::ContextProfile::kLong;
  else if (value == "xlong") profile = gem16gb::ContextProfile::kXlong;
  else if (value == "max") profile = gem16gb::ContextProfile::kMax;
  else return false;
  return true;
}

bool ParseKvStorage(std::string_view value, gem16gb::KvStorage& storage) {
  if (value == "shared") storage = gem16gb::KvStorage::kShared;
  else if (value == "separate") storage = gem16gb::KvStorage::kSeparate;
  else return false;
  return true;
}

int RunMemoryMode(int argc, char** argv) {
  std::filesystem::path model_directory;
  gem16gb::MemoryPlanOptions options;
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      model_directory = std::filesystem::path(argv[++index]);
    } else if (argument == "--profile" && index + 1 < argc) {
      if (!ParseProfile(argv[++index], options.context_profile)) {
        std::cerr << "error: unknown context profile\n";
        return 64;
      }
    } else if (argument == "--kv-storage" && index + 1 < argc) {
      if (!ParseKvStorage(argv[++index], options.kv_storage)) {
        std::cerr << "error: --kv-storage must be shared or separate\n";
        return 64;
      }
    } else if (argument == "--kv-element-bytes" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "1") options.kv_element_bytes = 1;
      else if (value == "2") options.kv_element_bytes = 2;
      else {
        std::cerr << "error: --kv-element-bytes must be 1 or 2\n";
        return 64;
      }
    } else {
      std::cerr << "error: unknown or incomplete memory option: " << argument << '\n';
      Usage(std::cerr);
      return 64;
    }
  }
  if (model_directory.empty()) {
    std::cerr << "error: memory mode requires --model\n";
    return 64;
  }

  auto plan = gem16gb::PlanCheckpointMemory(model_directory, options);
  if (!plan.ok()) {
    std::cerr << "error: " << plan.status().message() << '\n';
    return 1;
  }
  gem16gb::PrintMemoryPlanSummary(plan.value(), std::cerr);
  const auto write_status = gem16gb::WriteMemoryPlanJson(plan.value(), std::cout);
  if (!write_status.ok()) {
    std::cerr << "error: " << write_status.message() << '\n';
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::array modes = {
      "model-load", "prefill", "decode", "end-to-end", "kernel", "memory", "quality", "mtp"};
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    Usage(std::cout);
    return 0;
  }
  const std::string_view requested = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
  if (requested == "memory") return RunMemoryMode(argc, argv);

  bool known = false;
  for (const auto mode : modes) known = known || requested == mode;
  std::cerr << "{\"schema_version\":1,\"status\":\"not_implemented\",\"mode\":\""
            << (known ? requested : "unknown") << "\",\"fallbacks\":0}\n";
  return known ? 2 : 64;
}
