#include "cuda/nvfp4/sm120.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kElementsPerKBlock = 64;
constexpr std::uint64_t kRowsPerWarp = 8;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

Status Invalid(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Status(StatusCode::kInternal,
                std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                    cudaGetErrorString(error));
}

bool PositiveFinite(float value) {
  return std::isfinite(value) && value > 0.0F;
}

__device__ __forceinline__ std::uint32_t LoadU32(const std::uint8_t* source) {
  return *reinterpret_cast<const std::uint32_t*>(source);
}

__global__ void Sm120DirectProjectionKernel(const std::uint8_t* packed_activation_e2m1,
                                            const std::uint8_t* activation_scales_e4m3fn,
                                            const std::uint8_t* packed_weight_e2m1,
                                            const std::uint8_t* weight_scales_e4m3fn,
                                            float* output,
                                            std::uint64_t rows,
                                            std::uint64_t contracting_elements,
                                            float output_divisor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x & (kWarpSize - 1U);
  const std::uint64_t global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  if (global_warp >= row_tiles) return;

  const unsigned row_in_tile = lane >> 2U;
  const unsigned k_quarter = lane & 3U;
  const std::uint64_t source_row = global_warp * kRowsPerWarp + row_in_tile;
  const std::uint64_t packed_row_bytes = contracting_elements / 2U;
  const std::uint64_t scale_row_bytes = contracting_elements / 16U;
  const std::uint64_t k_blocks = contracting_elements / kElementsPerKBlock;

  float d0 = 0.0F;
  float d1 = 0.0F;
  float d2 = 0.0F;
  float d3 = 0.0F;
  constexpr std::uint16_t instruction_block_id = 0;
  constexpr std::uint16_t instruction_thread_id = 0;

  for (std::uint64_t k_block = 0; k_block < k_blocks; ++k_block) {
    const std::uint64_t activation_byte = k_block * 32U + k_quarter * 4U;
    const std::uint32_t a_first = LoadU32(packed_activation_e2m1 + activation_byte);
    const std::uint32_t a_second = LoadU32(packed_activation_e2m1 + activation_byte + 16U);

    std::uint32_t b_first = 0;
    std::uint32_t b_second = 0;
    std::uint32_t scale_b = 0;
    if (source_row < rows) {
      const std::uint64_t weight_byte = source_row * packed_row_bytes + k_block * 32U +
                                        static_cast<std::uint64_t>(k_quarter) * 4U;
      b_first = LoadU32(packed_weight_e2m1 + weight_byte);
      b_second = LoadU32(packed_weight_e2m1 + weight_byte + 16U);
      scale_b = LoadU32(weight_scales_e4m3fn + source_row * scale_row_bytes + k_block * 4U);
    }
    const std::uint32_t scale_a = LoadU32(activation_scales_e4m3fn + k_block * 4U);

    float next0 = 0.0F;
    float next1 = 0.0F;
    float next2 = 0.0F;
    float next3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, {%16, %17}, "
        "%15, {%16, %17};\n"
        : "=f"(next0), "=f"(next1), "=f"(next2), "=f"(next3)
        : "r"(a_first), "r"(a_first), "r"(a_second), "r"(a_second), "r"(b_first),
          "r"(b_second), "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(scale_a),
          "r"(scale_b), "h"(instruction_block_id), "h"(instruction_thread_id));
    d0 = next0;
    d1 = next1;
    d2 = next2;
    d3 = next3;
  }

  if (lane < 4U) {
    const std::uint64_t output_row = global_warp * kRowsPerWarp + lane * 2U;
    if (output_row < rows) output[output_row] = d0 / output_divisor;
    if (output_row + 1U < rows) output[output_row + 1U] = d1 / output_divisor;
  }
#else
  (void)packed_activation_e2m1;
  (void)activation_scales_e4m3fn;
  (void)packed_weight_e2m1;
  (void)weight_scales_e4m3fn;
  (void)output;
  (void)rows;
  (void)contracting_elements;
  (void)output_divisor;
#endif
}

}  // namespace

Status LaunchNvfp4Sm120DirectProjection(const std::uint8_t* packed_activation_e2m1,
                                        const std::uint8_t* activation_scales_e4m3fn,
                                        const std::uint8_t* packed_weight_e2m1,
                                        const std::uint8_t* weight_scales_e4m3fn,
                                        float* output,
                                        std::uint64_t rows,
                                        std::uint64_t contracting_elements,
                                        float activation_global_divisor,
                                        float weight_global_divisor,
                                        cudaStream_t stream) {
  if (packed_activation_e2m1 == nullptr || activation_scales_e4m3fn == nullptr ||
      packed_weight_e2m1 == nullptr || weight_scales_e4m3fn == nullptr || output == nullptr) {
    return Invalid("SM120 NVFP4 direct projection requires non-null device pointers");
  }
  if (rows == 0U || contracting_elements == 0U ||
      contracting_elements % kElementsPerKBlock != 0U) {
    return Invalid("SM120 NVFP4 direct projection requires positive dimensions and K divisible by 64");
  }
  if (!PositiveFinite(activation_global_divisor) || !PositiveFinite(weight_global_divisor)) {
    return Invalid("SM120 NVFP4 direct projection global divisors must be positive and finite");
  }
  const float output_divisor = activation_global_divisor * weight_global_divisor;
  if (!PositiveFinite(output_divisor)) {
    return Invalid("SM120 NVFP4 direct projection global-divisor product overflowed");
  }
  const std::uint64_t row_tiles = (rows + kRowsPerWarp - 1U) / kRowsPerWarp;
  const std::uint64_t blocks = (row_tiles + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > static_cast<std::uint64_t>(std::numeric_limits<unsigned>::max())) {
    return Invalid("SM120 NVFP4 direct projection grid exceeds CUDA limits");
  }
  Sm120DirectProjectionKernel<<<static_cast<unsigned>(blocks), kThreadsPerBlock, 0, stream>>>(
      packed_activation_e2m1, activation_scales_e4m3fn, packed_weight_e2m1,
      weight_scales_e4m3fn, output, rows, contracting_elements, output_divisor);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Status::Ok()
             : CudaFailure("launch direct-source SM120 NVFP4 projection", error);
}

}  // namespace gem16gb::internal
