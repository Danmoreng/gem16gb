#include <iostream>
#include <string_view>

#include "g4/engine.h"

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--print-kernel-capabilities") {
    g4::PrintKernelCapabilities(std::cout);
    return 0;
  }
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Usage: g4-run --print-kernel-capabilities\n"
              << "Inference is not implemented in the repository-initialization milestone.\n";
    return 0;
  }
  std::cerr << "error: inference is not implemented; no precision fallback was attempted\n";
  return 2;
}

