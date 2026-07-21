#include "test.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "model/safetensors.h"

namespace {

void WriteU64(std::ofstream& output, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<char>((value >> (index * 8U)) & 0xFFU);
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteSafetensors(const std::filesystem::path& path, std::string header, std::size_t payload_bytes) {
  while (header.size() % 8U != 0) header.push_back(' ');
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  WriteU64(output, header.size());
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::vector<char> payload(payload_bytes, '\0');
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

}  // namespace

void RunSafetensorsTests() {
  const auto root = std::filesystem::temp_directory_path() / ("g4-safetensors-test-" + std::to_string(getpid()));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directories(root);

  WriteSafetensors(root / "model.safetensors",
                   R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}})", 8);
  auto valid = g4::internal::LoadSafetensorsDirectory(root);
  G4_CHECK(valid.ok());
  if (valid.ok()) {
    G4_CHECK(valid.value().size() == 2);
    G4_CHECK(valid.value()[0].name == "a");
    G4_CHECK(valid.value()[1].length == 4);
  }

  WriteSafetensors(root / "model.safetensors",
                   R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}})", 6);
  G4_CHECK(!g4::internal::LoadSafetensorsDirectory(root).ok());

  std::filesystem::remove(root / "model.safetensors");
  {
    std::ofstream index(root / "model.safetensors.index.json");
    index << R"({"weight_map":{"a":"../escape.safetensors"}})";
  }
  G4_CHECK(!g4::internal::LoadSafetensorsDirectory(root).ok());

  const auto external = root.parent_path() / (root.filename().string() + "-external.safetensors");
  WriteSafetensors(external, R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", 4);
  {
    std::ofstream index(root / "model.safetensors.index.json", std::ios::trunc);
    index << R"({"weight_map":{"a":"linked.safetensors"}})";
  }
  std::filesystem::create_symlink(external, root / "linked.safetensors", error);
  G4_CHECK(!error);
  G4_CHECK(!g4::internal::LoadSafetensorsDirectory(root).ok());

  std::filesystem::remove_all(root, error);
  G4_CHECK(!error);
  std::filesystem::remove(external, error);
  G4_CHECK(!error);
}
