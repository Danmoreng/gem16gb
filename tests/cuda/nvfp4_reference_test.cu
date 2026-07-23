#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/nvfp4/reference.h"
#include "cuda/nvfp4/sm120.h"
#include "cuda/nvfp4/mlp.h"
#include "gem16gb/fp8.h"
#include "gem16gb/nvfp4.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << __FILE__ << ':' << line << ": check failed: " << expression << '\n';
    ++failures;
  }
}

#define CUDA_TEST_CHECK(expression) Check(static_cast<bool>(expression), #expression, __LINE__)

bool CudaOk(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorName(error) << ": "
            << cudaGetErrorString(error) << '\n';
  ++failures;
  return false;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    if (!CudaOk(cudaMalloc(&data_, elements * sizeof(T)), "cudaMalloc")) data_ = nullptr;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const { return static_cast<T*>(data_); }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* data_ = nullptr;
  std::size_t elements_ = 0;
};

void TestCudaIntrinsicConformanceAndProjection() {
  std::array<float, 16> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        gem16gb::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(index)) / 2.0F;
  }
  const auto host_quantized = gem16gb::nvfp4::QuantizeActivation(host_activation, 2.0F);
  CUDA_TEST_CHECK(host_quantized.ok());
  if (!host_quantized.ok()) return;

  DeviceBuffer<float> device_activation(host_activation.size());
  DeviceBuffer<std::uint8_t> device_packed(host_activation.size() / 2U);
  DeviceBuffer<std::uint8_t> device_scales(host_activation.size() / 16U);
  if (device_activation.get() == nullptr || device_packed.get() == nullptr ||
      device_scales.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), host_activation.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy activation to device")) {
    return;
  }

  const gem16gb::Status quantize_status =
      gem16gb::internal::LaunchNvfp4ReferenceActivationQuantization(
          device_activation.get(), device_packed.get(), device_scales.get(),
          host_activation.size(), 2.0F, nullptr);
  CUDA_TEST_CHECK(quantize_status.ok());
  if (!quantize_status.ok() || !CudaOk(cudaDeviceSynchronize(), "quantize synchronize")) return;

  std::array<std::uint8_t, 8> gpu_packed{};
  std::array<std::uint8_t, 1> gpu_scales{};
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_packed.data(), device_packed.get(), device_packed.bytes(),
                                    cudaMemcpyDeviceToHost),
                             "copy packed activation to host"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_scales.data(), device_scales.get(), device_scales.bytes(),
                                    cudaMemcpyDeviceToHost),
                             "copy activation scales to host"));
  CUDA_TEST_CHECK(std::equal(gpu_packed.begin(), gpu_packed.end(),
                             host_quantized.value().packed_e2m1.begin()));
  CUDA_TEST_CHECK(gpu_scales[0] == host_quantized.value().block_scales_e4m3fn[0]);

  constexpr std::array<std::uint8_t, 8> weight = {
      0x37U, 0xC1U, 0x53U, 0xA0U, 0xFBU, 0x5DU, 0xADU, 0xFEU,
  };
  constexpr std::array<std::uint8_t, 1> weight_scales = {0x61U};
  DeviceBuffer<std::uint8_t> device_weight(weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<float> device_output(1);
  if (device_weight.get() == nullptr || device_weight_scales.get() == nullptr ||
      device_output.get() == nullptr) {
    return;
  }
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(device_weight.get(), weight.data(), device_weight.bytes(),
                                    cudaMemcpyHostToDevice),
                             "copy weight to device"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                                    device_weight_scales.bytes(), cudaMemcpyHostToDevice),
                             "copy weight scales to device"));

  const gem16gb::Status projection_status = gem16gb::internal::LaunchNvfp4ReferenceProjection(
      device_packed.get(), device_scales.get(), device_weight.get(), device_weight_scales.get(),
      device_output.get(), 1, 16, 2.0F, 9600.0F, nullptr);
  CUDA_TEST_CHECK(projection_status.ok());
  if (!projection_status.ok() || !CudaOk(cudaDeviceSynchronize(), "projection synchronize")) {
    return;
  }

  float gpu_output = 0.0F;
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(&gpu_output, device_output.get(), sizeof(gpu_output),
                                    cudaMemcpyDeviceToHost),
                             "copy projection output to host"));
  const auto expected = gem16gb::nvfp4::ReferenceDotProduct(
      host_quantized.value(), weight, weight_scales, 9600.0F);
  CUDA_TEST_CHECK(expected.ok());
  if (expected.ok()) {
    CUDA_TEST_CHECK(std::fabs(static_cast<double>(gpu_output) - expected.value()) < 1.0e-6);
  }
}

void StoreNibble(std::vector<std::uint8_t>& packed, std::size_t row, std::size_t k,
                 std::size_t packed_row_bytes, std::uint8_t nibble) {
  std::uint8_t& byte = packed[row * packed_row_bytes + k / 2U];
  const unsigned shift = k % 2U == 0U ? 0U : 4U;
  byte = static_cast<std::uint8_t>(byte | static_cast<std::uint8_t>(nibble << shift));
}

void TestDirectSourceSm120Projection() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t k_size = 64;
  constexpr float activation_divisor = 2.0F;
  constexpr float weight_divisor = 4.0F;

  std::array<float, k_size> host_activation{};
  for (std::size_t k = 0; k < k_size; ++k) {
    host_activation[k] =
        gem16gb::nvfp4::DecodeE2M1(static_cast<std::uint8_t>(k & 0x0FU)) /
        activation_divisor;
  }
  const auto quantized =
      gem16gb::nvfp4::QuantizeActivation(host_activation, activation_divisor);
  CUDA_TEST_CHECK(quantized.ok());
  if (!quantized.ok()) return;

  std::vector<std::uint8_t> packed_weight(rows * k_size / 2U, 0U);
  std::vector<std::uint8_t> weight_scales(rows * k_size / 16U, 0x38U);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      const auto code = static_cast<std::uint8_t>((row * 3U + k * 5U) & 0x0FU);
      StoreNibble(packed_weight, row, k, k_size / 2U, code);
    }
  }

  DeviceBuffer<std::uint8_t> device_activation(quantized.value().packed_e2m1.size());
  DeviceBuffer<std::uint8_t> device_activation_scales(
      quantized.value().block_scales_e4m3fn.size());
  DeviceBuffer<std::uint8_t> device_weight(packed_weight.size());
  DeviceBuffer<std::uint8_t> device_weight_scales(weight_scales.size());
  DeviceBuffer<float> device_output(rows);
  if (device_activation.get() == nullptr || device_activation_scales.get() == nullptr ||
      device_weight.get() == nullptr || device_weight_scales.get() == nullptr ||
      device_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_activation.get(), quantized.value().packed_e2m1.data(),
                         device_activation.bytes(), cudaMemcpyHostToDevice),
              "copy native activation") ||
      !CudaOk(cudaMemcpy(device_activation_scales.get(),
                         quantized.value().block_scales_e4m3fn.data(),
                         device_activation_scales.bytes(), cudaMemcpyHostToDevice),
              "copy native activation scales") ||
      !CudaOk(cudaMemcpy(device_weight.get(), packed_weight.data(), device_weight.bytes(),
                         cudaMemcpyHostToDevice),
              "copy native weights") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), weight_scales.data(),
                         device_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy native weight scales")) {
    return;
  }

  const gem16gb::Status status = gem16gb::internal::LaunchNvfp4Sm120DirectProjection(
      device_activation.get(), device_activation_scales.get(), device_weight.get(),
      device_weight_scales.get(), device_output.get(), rows, k_size, activation_divisor,
      weight_divisor, nullptr);
  CUDA_TEST_CHECK(status.ok());
  if (!status.ok() || !CudaOk(cudaDeviceSynchronize(), "native projection synchronize")) return;

  std::array<float, rows> output{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost),
              "copy native projection output")) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const std::span<const std::uint8_t> weight_row(
        packed_weight.data() + row * k_size / 2U, k_size / 2U);
    const std::span<const std::uint8_t> scale_row(
        weight_scales.data() + row * k_size / 16U, k_size / 16U);
    const auto expected = gem16gb::nvfp4::ReferenceDotProduct(
        quantized.value(), weight_row, scale_row, weight_divisor);
    CUDA_TEST_CHECK(expected.ok());
    if (expected.ok()) {
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(output[row]) - expected.value()) < 1.0e-5);
    }
  }
}

float GeluTanhReference(float value) {
  constexpr float square_root_two_over_pi = 0.7978845608028654F;
  constexpr float cubic = 0.044715F;
  return 0.5F * value *
         (1.0F + std::tanh(square_root_two_over_pi *
                           (value + cubic * value * value * value)));
}

void TestMlpElementwiseBridge() {
  constexpr std::array<float, 9> gate = {
      -4.0F, -1.5F, -0.25F, -0.0F, 0.0F, 0.125F, 0.75F, 2.0F, 5.0F};
  constexpr std::array<float, 9> up = {
      0.5F, -2.0F, 3.0F, 4.0F, -5.0F, 1.25F, -0.75F, 2.5F, -0.125F};
  constexpr std::array<float, 9> residual = {
      1.0F, 0.5F, -0.5F, 2.0F, -2.0F, 0.25F, 0.0F, -1.0F, 3.0F};

  DeviceBuffer<float> device_gate(gate.size());
  DeviceBuffer<float> device_up(up.size());
  DeviceBuffer<float> device_product(gate.size());
  DeviceBuffer<float> device_residual(residual.size());
  DeviceBuffer<float> device_output(gate.size());
  if (device_gate.get() == nullptr || device_up.get() == nullptr ||
      device_product.get() == nullptr || device_residual.get() == nullptr ||
      device_output.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_gate.get(), gate.data(), device_gate.bytes(),
                         cudaMemcpyHostToDevice), "copy Gate") ||
      !CudaOk(cudaMemcpy(device_up.get(), up.data(), device_up.bytes(),
                         cudaMemcpyHostToDevice), "copy Up") ||
      !CudaOk(cudaMemcpy(device_residual.get(), residual.data(), device_residual.bytes(),
                         cudaMemcpyHostToDevice), "copy residual")) {
    return;
  }

  const auto product_status = gem16gb::internal::LaunchGeluTanhProduct(
      device_gate.get(), device_up.get(), device_product.get(), gate.size(), nullptr);
  CUDA_TEST_CHECK(product_status.ok());
  const auto residual_status = gem16gb::internal::LaunchAddResidual(
      device_product.get(), device_residual.get(), device_output.get(), gate.size(), nullptr);
  CUDA_TEST_CHECK(residual_status.ok());
  if (!product_status.ok() || !residual_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "MLP elementwise synchronize")) {
    return;
  }

  std::array<float, gate.size()> output{};
  if (!CudaOk(cudaMemcpy(output.data(), device_output.get(), device_output.bytes(),
                         cudaMemcpyDeviceToHost), "copy MLP elementwise output")) {
    return;
  }
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float expected = GeluTanhReference(gate[index]) * up[index] + residual[index];
    CUDA_TEST_CHECK(std::fabs(output[index] - expected) < 1.0e-6F);
  }
}

void TestFp8ReferenceAndDirectProjection() {
  constexpr std::size_t rows = 8;
  constexpr std::size_t k_size = 32;
  std::array<float, k_size> host_activation{};
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    host_activation[index] =
        static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.125F;
  }
  const auto host_quantized = gem16gb::fp8::QuantizeToken(host_activation);
  CUDA_TEST_CHECK(host_quantized.ok());
  if (!host_quantized.ok()) return;

  std::vector<std::uint8_t> host_weight(rows * k_size);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < k_size; ++k) {
      const float value = static_cast<float>(static_cast<int>((row * 5U + k * 3U) % 15U) - 7) /
                          4.0F;
      const auto encoded = gem16gb::fp8::EncodeE4M3Fn(value);
      CUDA_TEST_CHECK(encoded.ok());
      if (!encoded.ok()) return;
      host_weight[row * k_size + k] = encoded.value();
    }
  }
  constexpr std::array<std::uint16_t, rows> host_weight_scales = {
      0x3F80U, 0x3F00U, 0x3FC0U, 0x4000U, 0x3E80U, 0xBF80U, 0x4080U, 0x3F40U};
  // The checkpoint contract requires positive scales; keep the synthetic fixture valid too.
  std::array<std::uint16_t, rows> positive_weight_scales = host_weight_scales;
  positive_weight_scales[5] = 0x3E00U;

  DeviceBuffer<float> device_input(k_size);
  DeviceBuffer<std::uint8_t> device_activation(k_size);
  DeviceBuffer<float> device_activation_scale(1);
  DeviceBuffer<std::uint8_t> device_weight(host_weight.size());
  DeviceBuffer<std::uint16_t> device_weight_scales(rows);
  DeviceBuffer<float> device_reference(rows);
  DeviceBuffer<float> device_native(rows);
  if (device_input.get() == nullptr || device_activation.get() == nullptr ||
      device_activation_scale.get() == nullptr || device_weight.get() == nullptr ||
      device_weight_scales.get() == nullptr || device_reference.get() == nullptr ||
      device_native.get() == nullptr) {
    return;
  }
  if (!CudaOk(cudaMemcpy(device_input.get(), host_activation.data(), device_input.bytes(),
                         cudaMemcpyHostToDevice), "copy FP8 input") ||
      !CudaOk(cudaMemcpy(device_weight.get(), host_weight.data(), device_weight.bytes(),
                         cudaMemcpyHostToDevice), "copy FP8 weight") ||
      !CudaOk(cudaMemcpy(device_weight_scales.get(), positive_weight_scales.data(),
                         device_weight_scales.bytes(), cudaMemcpyHostToDevice),
              "copy FP8 weight scales")) {
    return;
  }

  const auto quantize_status = gem16gb::internal::LaunchFp8ReferenceTokenQuantization(
      device_input.get(), device_activation.get(), device_activation_scale.get(), k_size, nullptr);
  CUDA_TEST_CHECK(quantize_status.ok());
  if (!quantize_status.ok() || !CudaOk(cudaDeviceSynchronize(), "FP8 quantize synchronize")) {
    return;
  }
  std::array<std::uint8_t, k_size> gpu_activation{};
  float gpu_scale = 0.0F;
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(gpu_activation.data(), device_activation.get(),
                                    device_activation.bytes(), cudaMemcpyDeviceToHost),
                             "copy FP8 activation"));
  CUDA_TEST_CHECK(CudaOk(cudaMemcpy(&gpu_scale, device_activation_scale.get(), sizeof(float),
                                    cudaMemcpyDeviceToHost),
                             "copy FP8 activation scale"));
  CUDA_TEST_CHECK(gpu_scale == host_quantized.value().scale);
  CUDA_TEST_CHECK(std::equal(gpu_activation.begin(), gpu_activation.end(),
                             host_quantized.value().values_e4m3fn.begin()));

  const auto reference_status = gem16gb::internal::LaunchFp8ReferenceProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_reference.get(), rows, k_size, nullptr);
  const auto native_status = gem16gb::internal::LaunchFp8Sm120DirectProjection(
      device_activation.get(), device_activation_scale.get(), device_weight.get(),
      device_weight_scales.get(), device_native.get(), rows, k_size, nullptr);
  CUDA_TEST_CHECK(reference_status.ok());
  CUDA_TEST_CHECK(native_status.ok());
  if (!reference_status.ok() || !native_status.ok() ||
      !CudaOk(cudaDeviceSynchronize(), "FP8 projection synchronize")) {
    return;
  }
  std::array<float, rows> reference_output{};
  std::array<float, rows> native_output{};
  if (!CudaOk(cudaMemcpy(reference_output.data(), device_reference.get(), device_reference.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 reference output") ||
      !CudaOk(cudaMemcpy(native_output.data(), device_native.get(), device_native.bytes(),
                         cudaMemcpyDeviceToHost), "copy FP8 native output")) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const auto expected = gem16gb::fp8::ReferenceDotProduct(
        host_quantized.value(),
        std::span<const std::uint8_t>(host_weight.data() + row * k_size, k_size),
        positive_weight_scales[row]);
    CUDA_TEST_CHECK(expected.ok());
    if (expected.ok()) {
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(reference_output[row]) - expected.value()) <
                         1.0e-4);
      CUDA_TEST_CHECK(std::fabs(static_cast<double>(native_output[row]) - expected.value()) <
                         1.0e-4);
    }
  }
}

}  // namespace

int main() {
  int device_count = 0;
  if (!CudaOk(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count == 0) {
    std::cerr << "CUDA test requires one device\n";
    return 1;
  }
  TestCudaIntrinsicConformanceAndProjection();
  TestDirectSourceSm120Projection();
  TestMlpElementwiseBridge();
  TestFp8ReferenceAndDirectProjection();
  if (failures != 0) {
    std::cerr << failures << " CUDA test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all CUDA tests passed\n";
  return 0;
}
