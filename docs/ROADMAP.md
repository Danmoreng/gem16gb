# Roadmap

Current stage: Milestone 1, checkpoint inspector and loader.

## Active gate

- Capture tokens, selected hidden states, and logits from a trusted runtime that directly understands the pinned
  compressed checkpoint.
- Pin current upstream llama.cpp and prove or disprove closest-parity conversion for the same snapshot.
- Turn the verified text-only inventory into deterministic device-arena and context-profile plans after baseline
  setup is reproducible.

## Baseline gate

The llama.cpp benchmark is deliberately before engine kernel optimization, but after source-checkpoint validation:

1. The physical C++/Python manifest comparison must be exact. This is complete for all 1,389 tensors.
2. A trusted direct-load runtime must produce fixed token IDs and reference logits.
3. The pinned llama.cpp converter must emit a tensor mapping report for the same source revision. Current upstream
   is pinned, but its converter rejects the checkpoint's mixed FP8/NVFP4 groups; this gate is explicitly blocked,
   not skipped.
4. Only after native SM120 NVFP4 execution, GPU residency, and quality are proven may timings be labeled as a
   closest-parity or native-NVFP4 baseline.
5. Maintain a separate fastest-practical llama.cpp baseline even if exact mixed-format parity is impossible.

## Next milestones

1. Establish the trusted reference runtime.
2. Establish and pin the three llama.cpp baseline tiers.
3. Implement the memory planner and CPU reference operators with strict golden fixtures.
4. Implement an unfused correctness engine.
5. Add and disassemble native SM120a NVFP4 kernels.
6. Specialize attention, KV cache, decode fusion, and CUDA Graph replay.
7. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.

The detailed gates and ordering in `AGENTS.md` are authoritative.
