#include "cuda/layer/checkpoint_probe.h"

#include "cuda/fp8/reference.h"
#include "cuda/fp8/sm120.h"
#include "cuda/layer/reference.h"
#include "cuda/nvfp4/mlp.h"
#include "gem16gb/model.h"
#include "platform/mapped_file.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gem16gb::internal {
namespace {

constexpr std::uint64_t kHidden = 3840;
constexpr std::uint64_t kQueryHeads = 16;
constexpr std::uint64_t kKvHeads = 8;
constexpr std::uint64_t kHeadDimension = 256;
constexpr std::uint64_t kQueryElements = kQueryHeads * kHeadDimension;
constexpr std::uint64_t kKvElements = kKvHeads * kHeadDimension;
constexpr std::uint64_t kContextTokens = 32;
constexpr float kEpsilon = 1.0e-6F;
constexpr double kTheta = 10000.0;
constexpr char kBase[] = "model.language_model.layers.0.";

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status CudaFailure(const char* operation, cudaError_t error) {
  return Error(StatusCode::kInternal,
               std::string(operation) + ": " + cudaGetErrorName(error) + ": " +
                   cudaGetErrorString(error));
}

const TensorInfo* FindTensor(const ModelManifest& manifest, const std::string& name) {
  const auto found = std::find_if(manifest.tensors.begin(), manifest.tensors.end(),
                                  [&](const TensorInfo& tensor) { return tensor.name == name; });
  return found == manifest.tensors.end() ? nullptr : &*found;
}

Result<std::span<const std::uint8_t>> TensorBytes(const MappedFile& file,
                                                  const TensorInfo& tensor) {
  if (tensor.byte_offset > file.size() || tensor.byte_length > file.size() - tensor.byte_offset ||
      tensor.byte_length > std::numeric_limits<std::size_t>::max()) {
    return Error(StatusCode::kDataLoss, "tensor range is outside its shard: " + tensor.name);
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(file.data() + tensor.byte_offset);
  return std::span<const std::uint8_t>(begin, static_cast<std::size_t>(tensor.byte_length));
}

struct HostFp8Binding {
  std::span<const std::uint8_t> weight;
  std::span<const std::uint8_t> scales;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

Result<HostFp8Binding> BindFp8(const ModelManifest& manifest, const MappedFile& mapped,
                              const std::string& projection, std::uint64_t rows,
                              std::uint64_t contracting) {
  const std::string name = std::string(kBase) + "self_attn." + projection + "_proj.weight";
  const TensorInfo* weight = FindTensor(manifest, name);
  const TensorInfo* scales = FindTensor(manifest, name + "_scale");
  if (weight == nullptr || scales == nullptr) {
    return Error(StatusCode::kNotFound, "missing FP8 tensor family: " + name);
  }
  if (weight->shape != std::vector<std::uint64_t>{rows, contracting} ||
      scales->shape != std::vector<std::uint64_t>{rows, 1U} ||
      weight->storage_dtype != "F8_E4M3" || scales->storage_dtype != "BF16") {
    return Error(StatusCode::kDataLoss, "unexpected FP8 tensor geometry: " + name);
  }
  auto weight_bytes = TensorBytes(mapped, *weight);
  auto scale_bytes = TensorBytes(mapped, *scales);
  if (!weight_bytes.ok()) return weight_bytes.status();
  if (!scale_bytes.ok()) return scale_bytes.status();
  return HostFp8Binding{weight_bytes.value(), scale_bytes.value(), rows, contracting};
}

Result<std::span<const std::uint8_t>> BindBf16(const ModelManifest& manifest,
                                               const MappedFile& mapped,
                                               const std::string& name,
                                               std::uint64_t elements) {
  const TensorInfo* tensor = FindTensor(manifest, name);
  if (tensor == nullptr) return Error(StatusCode::kNotFound, "missing BF16 tensor: " + name);
  if (tensor->storage_dtype != "BF16" || tensor->byte_length != elements * 2U) {
    return Error(StatusCode::kDataLoss, "unexpected BF16 tensor geometry: " + name);
  }
  return TensorBytes(mapped, *tensor);
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { if (data_ != nullptr) (void)cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] Status Allocate(std::uint64_t elements, const char* label) {
    if (elements == 0U || elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return Error(StatusCode::kInvalidArgument, std::string(label) + " size is invalid");
    }
    elements_ = static_cast<std::size_t>(elements);
    const cudaError_t error = cudaMalloc(&data_, bytes());
    return error == cudaSuccess ? Status::Ok() : CudaFailure(label, error);
  }
  [[nodiscard]] T* get() const { return static_cast<T*>(data_); }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

 private:
  void* data_ = nullptr;
  std::size_t elements_ = 0;
};

struct DeviceFp8Binding {
  DeviceBuffer<std::uint8_t> weight;
  DeviceBuffer<std::uint16_t> scales;
  std::uint64_t rows = 0;
  std::uint64_t contracting = 0;
};

Status UploadBinding(const HostFp8Binding& host, DeviceFp8Binding& device) {
  device.rows = host.rows;
  device.contracting = host.contracting;
  Status status = device.weight.Allocate(host.weight.size(), "allocate FP8 weight");
  if (!status.ok()) return status;
  status = device.scales.Allocate(host.scales.size() / 2U, "allocate FP8 scales");
  if (!status.ok()) return status;
  cudaError_t error = cudaMemcpy(device.weight.get(), host.weight.data(), host.weight.size(),
                                 cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("copy FP8 weight", error);
  error = cudaMemcpy(device.scales.get(), host.scales.data(), host.scales.size(),
                     cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::Ok() : CudaFailure("copy FP8 scales", error);
}

std::vector<float> DeterministicHidden() {
  std::vector<float> values(kHidden);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double position = static_cast<double>(index);
    values[index] = static_cast<float>(0.21 * std::sin(position * 0.011) +
                                       0.13 * std::cos(position * 0.017) +
                                       static_cast<double>(static_cast<int>(index % 17U) - 8) * 0.003);
  }
  return values;
}

std::vector<float> DeterministicCache(bool value_cache) {
  std::vector<float> values(kContextTokens * kKvElements);
  for (std::uint64_t token = 0; token + 1U < kContextTokens; ++token) {
    for (std::uint64_t element = 0; element < kKvElements; ++element) {
      const double phase = static_cast<double>(token * 37U + element);
      values[static_cast<std::size_t>(token * kKvElements + element)] =
          static_cast<float>((value_cache ? 0.07 : 0.05) * std::sin(phase * 0.019) +
                             (value_cache ? 0.03 : 0.04) * std::cos(phase * 0.013));
    }
  }
  return values;
}

enum class ProjectionPath { kReference, kSm120 };

Status LaunchProjection(ProjectionPath path, const std::uint8_t* activation,
                        const float* activation_scale, const DeviceFp8Binding& binding,
                        float* output) {
  if (path == ProjectionPath::kReference) {
    return LaunchFp8ReferenceProjection(activation, activation_scale, binding.weight.get(),
                                        binding.scales.get(), output, binding.rows,
                                        binding.contracting, nullptr);
  }
  return LaunchFp8Sm120DirectProjection(activation, activation_scale, binding.weight.get(),
                                       binding.scales.get(), output, binding.rows,
                                       binding.contracting, nullptr);
}

struct PathBuffers {
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> q_norm;
  DeviceBuffer<float> k_norm;
  DeviceBuffer<float> v_norm;
  DeviceBuffer<float> key_cache;
  DeviceBuffer<float> value_cache;
  DeviceBuffer<float> scores;
  DeviceBuffer<float> attention;
  DeviceBuffer<std::uint8_t> o_activation;
  DeviceBuffer<float> o_scale;
  DeviceBuffer<float> o_output;
  DeviceBuffer<float> post_norm;
  DeviceBuffer<float> final;
};

Status AllocatePath(PathBuffers& path) {
  for (const Status status : {
      path.q.Allocate(kQueryElements, "allocate Q"),
      path.k.Allocate(kKvElements, "allocate K"),
      path.v.Allocate(kKvElements, "allocate V"),
      path.q_norm.Allocate(kQueryElements, "allocate normalized Q"),
      path.k_norm.Allocate(kKvElements, "allocate normalized K"),
      path.v_norm.Allocate(kKvElements, "allocate normalized V"),
      path.key_cache.Allocate(kContextTokens * kKvElements, "allocate K cache"),
      path.value_cache.Allocate(kContextTokens * kKvElements, "allocate V cache"),
      path.scores.Allocate(kQueryHeads * kContextTokens, "allocate attention scores"),
      path.attention.Allocate(kQueryElements, "allocate attention output"),
      path.o_activation.Allocate(kQueryElements, "allocate O activation"),
      path.o_scale.Allocate(1, "allocate O activation scale"),
      path.o_output.Allocate(kHidden, "allocate O projection output"),
      path.post_norm.Allocate(kHidden, "allocate post-attention norm"),
      path.final.Allocate(kHidden, "allocate attention residual"),
  }) if (!status.ok()) return status;
  return Status::Ok();
}

std::uint64_t PathBytes(const PathBuffers& path) {
  return static_cast<std::uint64_t>(
      path.q.bytes() + path.k.bytes() + path.v.bytes() + path.q_norm.bytes() +
      path.k_norm.bytes() + path.v_norm.bytes() + path.key_cache.bytes() +
      path.value_cache.bytes() + path.scores.bytes() + path.attention.bytes() +
      path.o_activation.bytes() + path.o_scale.bytes() + path.o_output.bytes() +
      path.post_norm.bytes() + path.final.bytes());
}

Status RunPath(ProjectionPath projection_path, PathBuffers& path,
               const DeviceFp8Binding& q_binding, const DeviceFp8Binding& k_binding,
               const DeviceFp8Binding& v_binding, const DeviceFp8Binding& o_binding,
               const std::uint8_t* input_activation, const float* input_scale,
               const float* residual, const std::uint16_t* q_norm_weight,
               const std::uint16_t* k_norm_weight, const std::uint16_t* post_norm_weight,
               const std::vector<float>& host_keys, const std::vector<float>& host_values) {
  cudaError_t error = cudaMemcpy(path.key_cache.get(), host_keys.data(), path.key_cache.bytes(),
                                 cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("initialize K cache", error);
  error = cudaMemcpy(path.value_cache.get(), host_values.data(), path.value_cache.bytes(),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("initialize V cache", error);
  for (const Status status : {
      LaunchProjection(projection_path, input_activation, input_scale, q_binding, path.q.get()),
      LaunchProjection(projection_path, input_activation, input_scale, k_binding, path.k.get()),
      LaunchProjection(projection_path, input_activation, input_scale, v_binding, path.v.get()),
      LaunchRmsNorm(path.q.get(), q_norm_weight, path.q_norm.get(), kQueryHeads,
                    kHeadDimension, kEpsilon, nullptr),
      LaunchRmsNorm(path.k.get(), k_norm_weight, path.k_norm.get(), kKvHeads,
                    kHeadDimension, kEpsilon, nullptr),
      LaunchRmsNorm(path.v.get(), nullptr, path.v_norm.get(), kKvHeads,
                    kHeadDimension, kEpsilon, nullptr),
      LaunchRotaryEmbedding(path.q_norm.get(), kQueryHeads, kHeadDimension, kHeadDimension,
                            kContextTokens - 1U, kTheta, nullptr),
      LaunchRotaryEmbedding(path.k_norm.get(), kKvHeads, kHeadDimension, kHeadDimension,
                            kContextTokens - 1U, kTheta, nullptr),
      LaunchAppendKv(path.k_norm.get(), path.v_norm.get(), path.key_cache.get(),
                     path.value_cache.get(), kContextTokens - 1U, kKvHeads,
                     kHeadDimension, nullptr),
      LaunchLocalAttentionDecode(path.q_norm.get(), path.key_cache.get(), path.value_cache.get(),
                                 path.scores.get(), path.attention.get(), kQueryHeads, kKvHeads,
                                 kHeadDimension, kContextTokens, nullptr),
      LaunchFp8ReferenceTokenQuantization(path.attention.get(), path.o_activation.get(),
                                          path.o_scale.get(), kQueryElements, nullptr),
      LaunchProjection(projection_path, path.o_activation.get(), path.o_scale.get(), o_binding,
                       path.o_output.get()),
      LaunchRmsNorm(path.o_output.get(), post_norm_weight, path.post_norm.get(), 1, kHidden,
                    kEpsilon, nullptr),
      LaunchAddResidual(path.post_norm.get(), residual, path.final.get(), kHidden, nullptr),
  }) if (!status.ok()) return status;
  return Status::Ok();
}

}  // namespace

Result<LocalAttentionCheckpointProbeResult> RunLayer0LocalAttentionCheckpointProbe(
    const std::filesystem::path& model_directory) {
  auto manifest = InspectCheckpoint({model_directory, true});
  if (!manifest.ok()) return manifest.status();
  auto mapped = MappedFile::Open(model_directory / "model.safetensors");
  if (!mapped.ok()) return mapped.status();
  auto q = BindFp8(manifest.value(), mapped.value(), "q", kQueryElements, kHidden);
  auto k = BindFp8(manifest.value(), mapped.value(), "k", kKvElements, kHidden);
  auto v = BindFp8(manifest.value(), mapped.value(), "v", kKvElements, kHidden);
  auto o = BindFp8(manifest.value(), mapped.value(), "o", kHidden, kQueryElements);
  auto input_norm = BindBf16(manifest.value(), mapped.value(), std::string(kBase) +
                             "input_layernorm.weight", kHidden);
  auto q_norm = BindBf16(manifest.value(), mapped.value(), std::string(kBase) +
                         "self_attn.q_norm.weight", kHeadDimension);
  auto k_norm = BindBf16(manifest.value(), mapped.value(), std::string(kBase) +
                         "self_attn.k_norm.weight", kHeadDimension);
  auto post_norm = BindBf16(manifest.value(), mapped.value(), std::string(kBase) +
                            "post_attention_layernorm.weight", kHidden);
  if (!q.ok()) return q.status(); if (!k.ok()) return k.status();
  if (!v.ok()) return v.status(); if (!o.ok()) return o.status();
  if (!input_norm.ok()) return input_norm.status(); if (!q_norm.ok()) return q_norm.status();
  if (!k_norm.ok()) return k_norm.status(); if (!post_norm.ok()) return post_norm.status();

  cudaDeviceProp properties{};
  cudaError_t error = cudaGetDeviceProperties(&properties, 0);
  if (error != cudaSuccess) return CudaFailure("cudaGetDeviceProperties", error);
  if (properties.major != 12 || properties.minor != 0) {
    return Error(StatusCode::kUnsupported, "local attention checkpoint probe requires SM120");
  }

  DeviceFp8Binding device_q, device_k, device_v, device_o;
  for (const Status status : {UploadBinding(q.value(), device_q), UploadBinding(k.value(), device_k),
                              UploadBinding(v.value(), device_v), UploadBinding(o.value(), device_o)})
    if (!status.ok()) return status;
  DeviceBuffer<float> hidden, normalized;
  DeviceBuffer<std::uint8_t> activation;
  DeviceBuffer<float> activation_scale;
  DeviceBuffer<std::uint16_t> input_norm_weight, q_norm_weight, k_norm_weight, post_norm_weight;
  for (const Status status : {
      hidden.Allocate(kHidden, "allocate hidden"), normalized.Allocate(kHidden, "allocate normalized hidden"),
      activation.Allocate(kHidden, "allocate FP8 hidden"), activation_scale.Allocate(1, "allocate hidden scale"),
      input_norm_weight.Allocate(kHidden, "allocate input norm weight"),
      q_norm_weight.Allocate(kHeadDimension, "allocate Q norm weight"),
      k_norm_weight.Allocate(kHeadDimension, "allocate K norm weight"),
      post_norm_weight.Allocate(kHidden, "allocate post norm weight")})
    if (!status.ok()) return status;
  const auto host_hidden = DeterministicHidden();
  const auto host_keys = DeterministicCache(false);
  const auto host_values = DeterministicCache(true);
  const std::array<std::pair<void*, std::span<const std::uint8_t>>, 4> norm_copies = {{
      {input_norm_weight.get(), input_norm.value()},
      {q_norm_weight.get(), q_norm.value()},
      {k_norm_weight.get(), k_norm.value()},
      {post_norm_weight.get(), post_norm.value()},
  }};
  for (const auto& copy : norm_copies) {
    error = cudaMemcpy(copy.first, copy.second.data(), copy.second.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return CudaFailure("copy norm weight", error);
  }
  error = cudaMemcpy(hidden.get(), host_hidden.data(), hidden.bytes(), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return CudaFailure("copy hidden", error);
  Status status = LaunchRmsNorm(hidden.get(), input_norm_weight.get(), normalized.get(), 1,
                                kHidden, kEpsilon, nullptr);
  if (!status.ok()) return status;
  status = LaunchFp8ReferenceTokenQuantization(normalized.get(), activation.get(),
                                               activation_scale.get(), kHidden, nullptr);
  if (!status.ok()) return status;

  PathBuffers reference, native;
  status = AllocatePath(reference); if (!status.ok()) return status;
  status = AllocatePath(native); if (!status.ok()) return status;
  status = RunPath(ProjectionPath::kReference, reference, device_q, device_k, device_v, device_o,
                   activation.get(), activation_scale.get(), hidden.get(), q_norm_weight.get(),
                   k_norm_weight.get(), post_norm_weight.get(), host_keys, host_values);
  if (!status.ok()) return status;
  status = RunPath(ProjectionPath::kSm120, native, device_q, device_k, device_v, device_o,
                   activation.get(), activation_scale.get(), hidden.get(), q_norm_weight.get(),
                   k_norm_weight.get(), post_norm_weight.get(), host_keys, host_values);
  if (!status.ok()) return status;
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return CudaFailure("local attention probe synchronize", error);
  std::vector<float> host_reference(kHidden), host_native(kHidden);
  error = cudaMemcpy(host_reference.data(), reference.final.get(), reference.final.bytes(), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy reference attention output", error);
  error = cudaMemcpy(host_native.data(), native.final.get(), native.final.bytes(), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return CudaFailure("copy native attention output", error);

  LocalAttentionCheckpointProbeResult result;
  result.context_tokens = kContextTokens;
  double squared_error = 0.0, dot = 0.0, reference_norm = 0.0, native_norm = 0.0;
  for (std::size_t index = 0; index < host_reference.size(); ++index) {
    if (!std::isfinite(host_reference[index]) || !std::isfinite(host_native[index]))
      return Error(StatusCode::kDataLoss, "local attention probe produced non-finite output");
    const double difference = static_cast<double>(host_reference[index]) - host_native[index];
    result.reference_native_max_abs = std::max(result.reference_native_max_abs, std::fabs(difference));
    squared_error += difference * difference;
    dot += static_cast<double>(host_reference[index]) * host_native[index];
    reference_norm += static_cast<double>(host_reference[index]) * host_reference[index];
    native_norm += static_cast<double>(host_native[index]) * host_native[index];
  }
  result.reference_native_rms = std::sqrt(squared_error / static_cast<double>(kHidden));
  result.reference_native_cosine = dot / std::sqrt(reference_norm * native_norm);
  for (const std::uint64_t element : std::array<std::uint64_t, 8>{0, 1, 127, 511, 1023, 2047, 3071, 3839})
    result.samples.push_back({element, host_reference[element], host_native[element]});
  result.device_bytes = static_cast<std::uint64_t>(
      device_q.weight.bytes() + device_q.scales.bytes() + device_k.weight.bytes() +
      device_k.scales.bytes() + device_v.weight.bytes() + device_v.scales.bytes() +
      device_o.weight.bytes() + device_o.scales.bytes() + hidden.bytes() + normalized.bytes() +
      activation.bytes() + activation_scale.bytes() + input_norm_weight.bytes() +
      q_norm_weight.bytes() + k_norm_weight.bytes() + post_norm_weight.bytes()) +
      PathBytes(reference) + PathBytes(native);
  return result;
}

}  // namespace gem16gb::internal
