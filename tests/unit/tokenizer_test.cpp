#include "gem16gb/tokenizer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test.h"

namespace {

void TestBpeEncodeDecodeAndSpecialTokens() {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("gem16gb-tokenizer-test-" + std::to_string(suffix));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  GEM16GB_CHECK(!error);
  const std::filesystem::path path = directory / "tokenizer.json";
  {
    std::ofstream output(path, std::ios::binary);
    output << R"({
      "added_tokens": [
        {"id": 0, "content": "<s>", "special": true}
      ],
      "model": {
        "type": "BPE",
        "byte_fallback": true,
        "vocab": {
          "<s>": 0,
          "a": 1,
          "b": 2,
          "ab": 3,
          "▁": 4,
          "▁ab": 5
        },
        "merges": [
          ["a", "b"],
          ["▁", "ab"]
        ]
      }
    })";
    GEM16GB_CHECK(output.good());
  }

  auto tokenizer = gem16gb::Tokenizer::Load(path);
  GEM16GB_CHECK(tokenizer.ok());
  if (tokenizer.ok()) {
    auto encoded = tokenizer.value().Encode("<s>ab ab");
    GEM16GB_CHECK(encoded.ok());
    if (encoded.ok()) {
      GEM16GB_CHECK(encoded.value() ==
                    std::vector<std::uint32_t>({0U, 3U, 5U}));
      auto decoded = tokenizer.value().Decode(encoded.value(), false);
      GEM16GB_CHECK(decoded.ok());
      if (decoded.ok()) GEM16GB_CHECK(decoded.value() == "<s>ab ab");
      auto without_special = tokenizer.value().Decode(encoded.value(), true);
      GEM16GB_CHECK(without_special.ok());
      if (without_special.ok()) GEM16GB_CHECK(without_special.value() == "ab ab");
    }
  }
  std::filesystem::remove(path, error);
  GEM16GB_CHECK(!error);
  std::filesystem::remove(directory, error);
  GEM16GB_CHECK(!error);
}

struct TokenizerTests {
  TokenizerTests() { TestBpeEncodeDecodeAndSpecialTokens(); }
} tokenizer_tests;

}  // namespace
