# Roadmap

Current stage: Milestone 1, checkpoint inspector and loader.

## Active gate

- Keep the verified 1,389-tensor manifest testable as loader classification becomes stricter.
- Add a byte-perfect comparison against an independent Python Safetensors reader.
- Validate the complete expected-role inventory for all 48 decoder layers.
- Turn the verified text-only inventory into deterministic device-arena and context-profile plans.

## Next milestones

1. Complete Level 0 loader integrity and byte-perfect Python comparisons.
2. Build CPU reference operators and strict golden fixtures.
3. Establish the trusted reference runtime and pinned llama.cpp baseline.
4. Implement an unfused correctness engine.
5. Add and disassemble native SM120a NVFP4 kernels.
6. Specialize attention, KV cache, decode fusion, and CUDA Graph replay.
7. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.

The detailed gates and ordering in `AGENTS.md` are authoritative.
