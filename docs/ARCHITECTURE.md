# Architecture

## Current loader path

`gem16gb-inspect` validates required checkpoint metadata, parses `config.json`, compiles its quantization target regexes
once, memory-maps Safetensors files, and builds a deterministic tensor manifest. No model payload is copied into
host RAM by the inspector.

The manifest is the only planned source for tensor names, shapes, offsets, dtype classes, scale relationships,
text-only residency, and tied-weight aliasing. Execution code must consume it rather than infer tensor inventory.

## Hardware backend boundary

`gem16gb` targets the approximately 16 GB NVIDIA CUDA GPU class. Architecture-specific kernels and dispatch live
behind explicit capability checks; the first implementation is Blackwell SM120/SM120a. Model execution plans,
allocator contracts, tensor manifests, and correctness fixtures must not encode a retail board name. A later CUDA
architecture backend should reuse those contracts while supplying its own kernels and measured dispatch choices.

## Planned execution split

Prefill and decode will use immutable, separate execution plans. Both will draw from named preallocated device
arenas. Decode will eventually use fixed addresses and CUDA Graph replay; it may not allocate, access files, or
compile code in the token loop.

The model-specific sequence is attention normalization and FP8 projections, specialized local/global attention,
then NVFP4 MLP projections and residual updates. This sequence now exists both as an independent Layer-0 comparison
probe and as an unfused, batch-one 48-layer greedy characterization. The latter loads the complete text-only model
into one aligned arena, keeps separate K/V state and reusable workspace allocations fixed for the run, applies the
tied BF16 embedding/output matrix, exact logit softcap, and GPU argmax, and performs no token-loop allocation. It is
not yet graph-captured or benchmark-qualified.

The first full-model path intentionally accepts token IDs and limits the contiguous correctness cache to 1,024
positions. Prompt tokens pass through the one-token decode plan as a temporary correctness bridge. A separate
parallel prefill graph, circular local cache, growing global cache, FP8 cache storage, tokenizer, and sampling plans
remain required production components.

## NVFP4 execution boundary

The NVFP4 MLP backend has three deliberately separate layers:

1. A platform-independent numeric contract and CPU oracle define E2M1, E4M3FN, compressed-tensors global-scale
   divisors, dynamic local activation quantization, and the observable output cast.
2. A loader-owned weight view prefers the source Safetensors layout directly. If measurement proves a final
   architecture-specific layout necessary, a streamed transformation may replace it without changing any
   quantized code or scale and without retaining a second device copy.
3. Operator-owned decode and prefill plans select only explicitly qualified implementations for an exact shape and
   token extent. A correctness route, packed SIMT/GEMV route, and native SM120a MMA route are distinct capabilities;
   none may silently stand in for another.

For the pinned checkpoint, Gate and Up have identical input and weight global divisors in all 48 layers. The native
decode plan may therefore quantize their shared input once, contract both matrices, and apply Gemma's GELU-tanh
product in one closed operator. Down performs its own dynamic-local quantization and may fuse its residual epilogue.
The attention projections remain a separate dynamic-FP8/per-channel-FP8 path.

## Memory-plan boundary

The first runtime component now converts parsed model metadata and the authoritative text-only manifest into a
deterministic 256-byte-aligned base arena. It places immutable weights/model state, scales, and the selected KV
payload in named regions with checked offsets. The required separate K/V size and a diagnostic one-state lower
bound are retained in every result; shared physical cache selection is rejected.

The greedy characterization measures a 1,465,856-byte execution workspace containing hidden-state ping-pong,
quantized activations and scales, projection intermediates, attention scores, full logits, and GPU argmax state.
Its short-context BF16-semantics cache is a separate measured allocation. The general planner remains conservative
until production prefill, FP8 cache, graph, and sampling shapes are defined.
