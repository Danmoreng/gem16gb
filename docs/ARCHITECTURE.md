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
then NVFP4 MLP projections and residual updates. The tied BF16 embedding/output matrix and exact logit softcap are
separate critical paths. No executable inference graph exists yet.
