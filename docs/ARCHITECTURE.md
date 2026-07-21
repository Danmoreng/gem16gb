# Architecture

## Current loader path

`g4-inspect` validates required checkpoint metadata, parses `config.json`, compiles its quantization target regexes
once, memory-maps Safetensors files, and builds a deterministic tensor manifest. No model payload is copied into
host RAM by the inspector.

The manifest is the only planned source for tensor names, shapes, offsets, dtype classes, scale relationships,
text-only residency, and tied-weight aliasing. Execution code must consume it rather than infer tensor inventory.

## Planned execution split

Prefill and decode will use immutable, separate execution plans. Both will draw from named preallocated device
arenas. Decode will eventually use fixed addresses and CUDA Graph replay; it may not allocate, access files, or
compile code in the token loop.

The model-specific sequence is attention normalization and FP8 projections, specialized local/global attention,
then NVFP4 MLP projections and residual updates. The tied BF16 embedding/output matrix and exact logit softcap are
separate critical paths. No executable inference graph exists yet.

