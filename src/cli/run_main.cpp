#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gem16gb/engine.h"

namespace {

bool ParseUnsigned(std::string_view text, std::uint64_t& value) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ParseTokenIds(std::string_view text, std::vector<std::uint32_t>& tokens) {
  if (text.empty()) return false;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    std::uint64_t token = 0;
    if (end == begin || !ParseUnsigned(text.substr(begin, end - begin), token) ||
        token > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    tokens.push_back(static_cast<std::uint32_t>(token));
    if (comma == std::string_view::npos) break;
    begin = comma + 1U;
  }
  return !tokens.empty();
}

void PrintUsage() {
  std::cout
      << "Usage:\n"
      << "  gem16gb-run --print-kernel-capabilities\n"
      << "  gem16gb-run --model <checkpoint> --input-token-ids <id,id,...>\n"
      << "              [--max-tokens N] [--max-context N] --greedy\n"
      << "\nThe inference path is a correctness characterization and is not benchmark-qualified.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--print-kernel-capabilities") {
    gem16gb::PrintKernelCapabilities(std::cout);
    return 0;
  }
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    PrintUsage();
    return 0;
  }

  gem16gb::GreedyInferenceOptions options;
  bool greedy = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--model" && index + 1 < argc) {
      options.model_directory = argv[++index];
    } else if (argument == "--input-token-ids" && index + 1 < argc) {
      if (!ParseTokenIds(argv[++index], options.input_token_ids)) {
        std::cerr << "error: --input-token-ids must be a comma-separated unsigned list\n";
        return 64;
      }
    } else if (argument == "--max-tokens" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_generated_tokens)) {
        std::cerr << "error: --max-tokens must be an unsigned integer\n";
        return 64;
      }
    } else if (argument == "--max-context" && index + 1 < argc) {
      if (!ParseUnsigned(argv[++index], options.max_context_tokens)) {
        std::cerr << "error: --max-context must be an unsigned integer\n";
        return 64;
      }
    } else if (argument == "--greedy") {
      greedy = true;
    } else {
      std::cerr << "error: unknown or incomplete option: " << argument << '\n';
      return 64;
    }
  }
  if (!greedy) {
    std::cerr << "error: the initial inference path requires explicit --greedy\n";
    return 64;
  }
  auto result = gem16gb::RunGreedyInference(options);
  if (!result.ok()) {
    std::cerr << "error: " << result.status().message()
              << "; no precision fallback was attempted\n";
    return 2;
  }
  const gem16gb::Status write = gem16gb::WriteGreedyInferenceJson(result.value(), std::cout);
  if (!write.ok()) {
    std::cerr << "error: " << write.message() << '\n';
    return 1;
  }
  return 0;
}
