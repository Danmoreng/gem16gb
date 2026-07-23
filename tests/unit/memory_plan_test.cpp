#include "test.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include "gem16gb/memory.h"
#include "runtime/memory_plan.h"
#include "util/json.h"

namespace {

gem16gb::internal::ModelConfig PrimaryMemoryConfig() {
  gem16gb::internal::ModelConfig config;
  config.layer_count = 48;
  config.max_positions = 262144;
  config.sliding_window = 1024;
  config.local_kv_heads = 8;
  config.global_kv_heads = 1;
  config.local_head_dimension = 256;
  config.global_head_dimension = 512;
  config.attention_k_eq_v = true;
  for (std::uint64_t layer = 0; layer < config.layer_count; ++layer) {
    config.layer_types.push_back(layer % 6U == 5U ? "full_attention" : "sliding_attention");
  }
  return config;
}

gem16gb::ModelManifest PrimaryMemoryManifest() {
  gem16gb::ModelManifest manifest;
  manifest.tensors = {
      {.name = "weights", .quantization_class = "BF16", .byte_length = 8'668'020'512},
      {.name = "fp8_scales", .quantization_class = "FP8_WEIGHT_SCALE", .byte_length = 1'163'264},
      {.name = "nvfp4_local_scales",
       .quantization_class = "NVFP4_LOCAL_SCALE_E4M3",
       .byte_length = 530'841'600},
      {.name = "nvfp4_global_scales", .quantization_class = "NVFP4_GLOBAL_SCALE", .byte_length = 576},
      {.name = "nvfp4_input_scales", .quantization_class = "NVFP4_INPUT_SCALE", .byte_length = 576},
      {.name = "multimodal", .quantization_class = "BF16", .byte_length = 104'759'808,
       .loaded_in_text_only_mode = false},
  };
  manifest.text_only_tensor_bytes = 9'200'026'528;
  manifest.skipped_tensor_bytes = 104'759'808;
  return manifest;
}

}  // namespace

void RunMemoryPlanTests() {
  using gem16gb::internal::BuildMemoryPlan;
  using gem16gb::internal::ContextTokens;
  using gem16gb::ContextProfile;
  using gem16gb::KvStorage;
  using gem16gb::MemoryPlanOptions;

  GEM16GB_CHECK(ContextTokens(ContextProfile::kInteractive).value() == 8192);
  GEM16GB_CHECK(ContextTokens(ContextProfile::kStandard).value() == 32768);
  GEM16GB_CHECK(ContextTokens(ContextProfile::kLong).value() == 65536);
  GEM16GB_CHECK(ContextTokens(ContextProfile::kXlong).value() == 131072);
  GEM16GB_CHECK(ContextTokens(ContextProfile::kMax).value() == 262144);

  const auto config = PrimaryMemoryConfig();
  const auto manifest = PrimaryMemoryManifest();
  const MemoryPlanOptions shared_options{
      .context_profile = ContextProfile::kLong,
      .kv_storage = KvStorage::kShared,
  };
  auto shared = BuildMemoryPlan(config, manifest, shared_options);
  GEM16GB_CHECK(shared.ok());
  if (shared.ok()) {
    GEM16GB_CHECK(shared.value().context_profile == "long");
    GEM16GB_CHECK(shared.value().kv_storage == "shared");
    GEM16GB_CHECK(shared.value().kv_element_bytes == 1);
    GEM16GB_CHECK(shared.value().arena_alignment == 256);
    GEM16GB_CHECK(shared.value().local_layer_count == 40);
    GEM16GB_CHECK(shared.value().global_layer_count == 8);
    GEM16GB_CHECK(shared.value().model_weight_bytes == 8'668'020'512);
    GEM16GB_CHECK(shared.value().scale_bytes == 532'006'016);
    GEM16GB_CHECK(shared.value().text_only_source_bytes == 9'200'026'528);
    GEM16GB_CHECK(shared.value().local_shared_kv_bytes == 83'886'080);
    GEM16GB_CHECK(shared.value().global_shared_kv_bytes == 268'435'456);
    GEM16GB_CHECK(shared.value().shared_kv_bytes == 352'321'536);
    GEM16GB_CHECK(shared.value().separate_kv_bytes == 704'643'072);
    GEM16GB_CHECK(shared.value().selected_kv_bytes == shared.value().shared_kv_bytes);
    GEM16GB_CHECK(shared.value().regions.size() == 3);
    for (const auto& region : shared.value().regions) {
      GEM16GB_CHECK(region.offset % 256U == 0);
      GEM16GB_CHECK(region.alignment == 256);
    }
    GEM16GB_CHECK(shared.value().total_arena_bytes % 256U == 0);
    GEM16GB_CHECK(shared.value().total_arena_bytes ==
                  shared.value().text_only_source_bytes + shared.value().selected_kv_bytes +
                      shared.value().padding_bytes);
    GEM16GB_CHECK(!shared.value().execution_workspaces_planned);

    std::ostringstream json_output;
    GEM16GB_CHECK(gem16gb::WriteMemoryPlanJson(shared.value(), json_output).ok());
    auto parsed_json = gem16gb::json::Parse(json_output.str());
    GEM16GB_CHECK(parsed_json.ok());
    if (parsed_json.ok()) {
      const auto* total_arena = parsed_json.value().find("total_arena_bytes");
      const auto* workspaces_planned = parsed_json.value().find("execution_workspaces_planned");
      const auto* fallbacks = parsed_json.value().find("fallbacks");
      GEM16GB_CHECK(total_arena != nullptr);
      GEM16GB_CHECK(workspaces_planned != nullptr);
      GEM16GB_CHECK(fallbacks != nullptr);
      if (total_arena != nullptr) GEM16GB_CHECK(total_arena->as_integer() == 9'552'348'416);
      if (workspaces_planned != nullptr) GEM16GB_CHECK(!workspaces_planned->as_bool());
      if (fallbacks != nullptr) GEM16GB_CHECK(fallbacks->as_integer() == 0);
    }
  }

  auto separate_options = shared_options;
  separate_options.kv_storage = KvStorage::kSeparate;
  auto separate = BuildMemoryPlan(config, manifest, separate_options);
  GEM16GB_CHECK(separate.ok());
  if (separate.ok()) {
    GEM16GB_CHECK(separate.value().selected_kv_bytes == 704'643'072);
    GEM16GB_CHECK(separate.value().total_arena_bytes - shared.value().total_arena_bytes == 352'321'536);
  }

  auto bf16_options = shared_options;
  bf16_options.kv_element_bytes = 2;
  auto bf16 = BuildMemoryPlan(config, manifest, bf16_options);
  GEM16GB_CHECK(bf16.ok());
  if (bf16.ok()) {
    GEM16GB_CHECK(bf16.value().shared_kv_bytes == 704'643'072);
  }

  GEM16GB_CHECK(!BuildMemoryPlan(config, manifest, MemoryPlanOptions{}).ok());

  auto invalid_alignment = shared_options;
  invalid_alignment.arena_alignment = 192;
  GEM16GB_CHECK(!BuildMemoryPlan(config, manifest, invalid_alignment).ok());

  auto non_shared_config = config;
  non_shared_config.attention_k_eq_v = false;
  GEM16GB_CHECK(!BuildMemoryPlan(non_shared_config, manifest, shared_options).ok());
  GEM16GB_CHECK(BuildMemoryPlan(non_shared_config, manifest, separate_options).ok());

  auto short_config = config;
  short_config.max_positions = 32768;
  GEM16GB_CHECK(!BuildMemoryPlan(short_config, manifest, shared_options).ok());

  auto inconsistent_manifest = manifest;
  ++inconsistent_manifest.text_only_tensor_bytes;
  GEM16GB_CHECK(!BuildMemoryPlan(config, inconsistent_manifest, shared_options).ok());

  auto overflow_options = shared_options;
  overflow_options.kv_element_bytes = std::numeric_limits<std::uint64_t>::max();
  GEM16GB_CHECK(!BuildMemoryPlan(config, manifest, overflow_options).ok());
}
