#include <array>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  constexpr std::array modes = {
      "model-load", "prefill", "decode", "end-to-end", "kernel", "memory", "quality", "mtp"};
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Usage: g4-bench <model-load|prefill|decode|end-to-end|kernel|memory|quality|mtp> [options]\n";
    return 0;
  }
  const std::string_view requested = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
  bool known = false;
  for (const auto mode : modes) known = known || requested == mode;
  std::cerr << "{\"schema_version\":1,\"status\":\"not_implemented\",\"mode\":\"" << (known ? requested : "unknown")
            << "\",\"fallbacks\":0}\n";
  return known ? 2 : 64;
}
