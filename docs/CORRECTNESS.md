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
the compact source weight and scale layouts directly. Its synthetic eight-row/64-K output and all three real
Layer-0 MLP projection shapes match the same oracle. The real Gate, Up, and Down maximum CUDA-reference/native
absolute differences were `1.1920929e-7`, `5.9604645e-8`, and `0`, respectively.

The first complete Layer-0 MLP characterization now executes input quantization, Gate and Up, Gemma GELU-tanh
product, Down-input quantization, Down, and residual addition without a host round trip. CPU/CUDA quantized input
bytes match exactly at both quantization boundaries. For its deterministic fixture the native and CUDA-reference
Down-input bytes and all 3,840 final float values also match exactly; eight final rows match the binary64 Down
oracle plus residual within `6.7374888e-9`. This remains a deterministic characterization, not a substitute for
the pinned hidden-state golden distribution.

Disassembly of the CUDA test binary contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. This is native-instruction and
real-shape evidence, but the kernel remains experimental until layer-golden, numerical-distribution, memory-arena,
and end-to-end gates pass.

Level 1 FP8 bring-up now includes host E4M3FN/BF16 decoding, dynamic per-token activation quantization, a binary64
per-channel-scale projection oracle, an independent CUDA scalar reference, and a direct-source SM120 tensor-core
route. The real Layer-0 Q `[4096,3840]`, K/V `[2048,3840]`, and O `[3840,4096]` shapes all produce bit-identical
CPU/CUDA activation bytes and scale bits. Across those fixtures the largest CUDA-reference/native absolute
difference is `8.9406967e-7`; the largest selected-row binary64/native difference is below `1.0e-7`.
Disassembly additionally contains `QMMA.16832.F32.E4M3.E4M3`. As with NVFP4, this proves the intended arithmetic
instruction and real storage mapping.

The first unfused local-attention checkpoint characterization now executes the real Layer-0 BF16 norm tensors and
FP8 Q/K/V/O tensor families through input RMSNorm, per-head Q/K normalization, scale-free V normalization, local
RoPE at position 31, K/V append/read, grouped-query causal attention over a deterministic 32-token cache, FP32
softmax, O projection, post-attention RMSNorm, and the residual update. The direct SM120 and independent CUDA scalar
projection routes produce a final 3,840-element maximum absolute difference of `4.8398972e-5`, RMS difference
`4.2503101e-6`, and cosine similarity `0.9999999999984577`. This validates operator composition and real tensor
binding without a persistent repack; it is not yet a trusted-runtime hidden-state golden or a performance result.

The corresponding full-attention characterization now executes the real Layer-5 tensor family with 16 query
heads, one 512-dimensional KV head, and proportional RoPE. It reuses the raw K projection as the V input because
the checkpoint has no `v_proj`, then deliberately diverges the states: K receives learned RMSNorm and proportional
RoPE over the first 25% of rotary frequencies, while V receives scale-free RMSNorm and no RoPE. Over the same
deterministic 32-token fixture, direct SM120 projections and the independent CUDA scalar route produce final
maximum absolute error `4.5299530e-6`, RMS error `5.5268314e-7`, and cosine similarity
`0.9999999999999085`. This also proves that `attention_k_eq_v` permits projection reuse but not shared physical
storage of the final K/V cache.

The complete Layer-0 decoder characterization now keeps the local-attention residual on device and continues
through pre-feedforward RMSNorm, dynamic NVFP4 input quantization, Gate/Up, GELU-tanh product, Down-input
quantization, Down, post-feedforward RMSNorm, the second residual update, and `layer_scalar`. It binds all real
Layer-0 FP8, NVFP4, BF16 norm, and scalar tensors directly from the checkpoint. The CUDA scalar-projection and
direct SM120 routes produce zero differing bytes at both NVFP4 activation boundaries. Their final 3,840-element
outputs have maximum absolute difference `4.7683716e-6`, RMS difference `2.8454761e-7`, and cosine similarity
`0.9999999999999643`. The probe owns 148,639,086 device bytes because it deliberately retains two complete
execution paths for comparison; this is not a production workspace or peak-VRAM estimate.

`tools/validate_layer_checkpoint.py` runs the local-attention, full-attention, and complete-decoder probes and
exports one combined JSON record. It enforces structural correctness gates but intentionally applies no model-wide
numeric tolerance. The next trusted fixture must use a real token sequence and contain the Layer-0 input, resulting
Layer-0 output, and matching K/V state needed to reproduce the selected decode position. The current deterministic
synthetic-cache characterization cannot honestly be compared directly with a prompt-derived hidden state.

The full-model greedy characterization now has two explicit generation gates. For
`exact_blue_no_thinking`, the checkpoint tokenizer and exact `chat_template.jinja` produce the committed 20 prompt
IDs, and the engine matches vLLM's complete `[9503, 106]` response (`blue<turn|>`). For the 31-token
`sky_sentence_no_thinking` reference, the engine matches `[818, 7217]` and diverges at generated step 2: vLLM
selects token `563`, while the engine selects vLLM's rank-2 token `7412`. This is a blocking correctness failure,
not an accepted numerical tolerance.

`--dump-logits` captures every selected position as full-vocabulary raw little-endian float32 after preallocating
host storage, and `tools/compare_logits.py` compares it with the committed vLLM top-20 distributions. At the first
divergent position, the reference token `563` is engine rank 2 with engine log probability `-1.56273` versus vLLM
`-0.06586`; token `7412` is engine rank 1 with `-0.23527` versus vLLM `-2.75336`. The discrepancy is too large to
attribute to an argmax tie. Full-vocabulary vLLM vectors and prompt-derived hidden/KV states remain necessary to
locate the earliest failing layer.

The native C++ tokenizer/template path reproduces all three committed reference prompt-ID sequences exactly:
20 tokens for exact-blue, 23 for the sky sentence, and 27 for the thinking arithmetic prompt. The application reads
the actual template file and accepts only the pinned supported revision. Its renderer currently supports
system/developer, user, and assistant text roles; tool calls and multimodal content fail visibly until their native
template branches are implemented. A separate German/Unicode probe containing umlauts, `ß`, and an emoji also
matches the Transformers tokenizer exactly across all 27 prompt IDs.

The patched same-source llama.cpp candidate supplies an independent comparison despite mapping FP8 attention
weights to BF16. It matches 50/65 reference output tokens overall: exact-blue is 2/2, the sky answer matches its
first 18 tokens before diverging, and the thinking trace matches 28/32. At our first sky divergence, both vLLM and
llama.cpp choose token `563`; our engine chooses token `7412`. The engine top-20 overlap is 13/20 against vLLM and
15/20 against llama.cpp at that step. Cross-engine bit identity is not required, but agreement between both
independent references and their nontrivial top-1 margin makes this early deviation a correctness investigation,
not an accepted implementation difference.

Reproduce the instruction check with:

```bash
python tools/verify_sm120_sass.py build/<OS>/blackwell-release/bin/gem16gb-cuda-tests
```

## Not yet established

Broad projection distributions, layer tolerances, full-vocabulary reference logits, trusted-runtime hidden-state
comparisons, broad cross-engine generation agreement, and task quality have not been measured. Therefore `tests/tolerances.yaml` is
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
