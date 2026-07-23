#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "gem16gb/status.h"

namespace gem16gb::internal {

struct Sm120Nvfp4SourceLayout {
  std::uint64_t rows = 0;
  std::uint64_t contracting_elements = 0;
  std::uint64_t row_tiles = 0;
  std::uint64_t k_blocks = 0;
  std::uint64_t packed_weight_bytes = 0;
  std::uint64_t scale_bytes = 0;
  std::uint64_t persistent_repack_bytes = 0;
};

struct Sm120Nvfp4WeightLaneFragment {
  std::array<std::uint32_t, 2> packed_e2m1{};
  std::uint32_t packed_e4m3fn_scales = 0;
  std::uint64_t source_row = 0;
  bool active = false;
};

[[nodiscard]] Result<Sm120Nvfp4SourceLayout> PlanSm120Nvfp4SourceLayout(
    std::uint64_t rows,
    std::uint64_t contracting_elements);

// Host oracle for the direct SM120 lane mapping. Production CUDA code performs the same two
// little-endian 32-bit source loads and one four-byte scale load directly from device-resident
// checkpoint storage.
[[nodiscard]] Result<Sm120Nvfp4WeightLaneFragment> LoadSm120Nvfp4WeightLaneFragment(
    const Sm120Nvfp4SourceLayout& layout,
    std::span<const std::uint8_t> packed_weight_e2m1,
    std::span<const std::uint8_t> weight_scales_e4m3fn,
    std::uint64_t row_tile,
    std::uint64_t k_block,
    std::uint32_t lane);

}  // namespace gem16gb::internal
