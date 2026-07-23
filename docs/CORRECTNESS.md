# Correctness

## Implemented level

Level 0 currently covers bounded JSON parsing, duplicate-key rejection, little-endian Safetensors header lengths,
shape-product overflow, known dtype byte sizes, payload bounds, exact byte lengths, overlapping ranges, duplicate
tensors across shards, index agreement, UTF-8 strings, shard path traversal rejection, and symlink escape rejection.

`gem16gb-inspect --validate` additionally checks the expected primary architecture dimensions and quantization mode,
then requires each classified NVFP4 packed weight to have local, global, and input scale tensors.

An independent Python reader compares the raw Safetensors headers against the exported C++ manifest. For pinned
revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, all 1,389 tensors match across physical shape, dtype, absolute
offset, byte length, shard, and alignment; total tensor payload is 9,304,786,336 bytes with zero mismatches. The
validated decoder inventory contains 29 tensors in every sliding-attention layer and 27 in every full-attention
layer; full-attention layers omit separate `v_proj` weight and scale tensors.

Level 1 NVFP4 bring-up now includes a platform-independent E2M1 and E4M3FN codec, round-to-nearest-even host
encoding, dynamic-local activation quantization in groups of 16, compressed-tensors global-divisor application, and
a binary64 W4A4 projection oracle. Tests exhaustively round-trip all finite E4M3FN words and all E2M1 nibbles,
exercise rounding, saturation, and error behavior, and pin the first 16 packed values and first local scale from
layer 0 Gate row 0 of the locked checkpoint.

The CUDA correctness route independently uses CUDA 13.3 FP4/FP8 conversion types, matches the host packed
activation and scale bytes, and matches the host projection oracle. A separate experimental SM120a kernel consumes
the compact source weight and scale layouts directly; its synthetic eight-row/64-K output matches the same oracle.
Disassembly of the CUDA test binary contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. This is instruction-path
evidence only, not yet a real-shape or production-kernel qualification.

Reproduce the instruction check with:

```bash
python tools/verify_sm120_sass.py build/<OS>/blackwell-release/bin/gem16gb-cuda-tests
```

## Not yet established

Real-shape projection distributions, layer tolerances, full-vocabulary logits, hidden-state comparisons,
cross-engine generation agreement, and task quality have not been measured. Therefore `tests/tolerances.yaml` is
intentionally empty. The committed vLLM
fixture provides greedy token IDs and top-20 log probabilities, but it is not a substitute for full-logit Level 3
metrics. Tolerances will be added only after reference distributions exist.

## Direct reference runtime

The checkpoint model card's reference recipe is used in a separate, ignored Python 3.13 environment. The first
validated environment contains vLLM 0.25.1, PyTorch 2.11.0+cu130, Transformers 5.14.1, compressed-tensors 0.17.0,
FlashInfer 0.6.13, and NVIDIA CUTLASS DSL 4.5.2. `tools/generate_golden.py` runs the locked local checkpoint with
network access disabled, batch one, an 8K context limit, eager execution, no prefix cache, no CPU offload, and all
multimodal limits set to zero. Model-supported chunked prefill remains enabled. It records exact templated prompt
IDs, greedy output IDs, and the top 20 log
probabilities at every generated position.

Reference-runtime startup logs are part of the evidence: vLLM must select `CutlassFP8ScaledMMLinearKernel` for the
attention projections and `FlashInferCutlassNvFp4LinearKernel` for the NVFP4 MLPs. Package selection alone does not
replace later per-kernel profiling.

Two consecutive runs on 2026-07-21 produced exactly identical prompt IDs, output IDs, and log probabilities. The
first engine initialization took 119.44 seconds while compiling and autotuning; the warm-cache initialization took
4.69 seconds. FlashInfer reported OOM for some autotuning tactics and stored default fallbacks for those shapes.
This does not invalidate the correctness fixture, but it disqualifies these runs as performance evidence and must
be revisited when configuring any vLLM speed baseline.

Run with the reference environment activated so its `ninja` executable is visible:

```bash
PATH="$PWD/third_party/cache/unsloth-nvfp4-env/bin:$PATH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  third_party/cache/unsloth-nvfp4-env/bin/python tools/generate_golden.py \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --output tests/golden/vllm-gemma4-12b-nvfp4.json
```

Reproduce the physical manifest comparison with:

```bash
build/host-debug/bin/gem16gb-inspect --model <checkpoint> --validate --json build/manifest.json
python3 tools/compare_manifests.py --model <checkpoint> --manifest build/manifest.json
```
