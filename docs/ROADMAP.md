# Roadmap

Current stage: Milestone 1, checkpoint inspector and loader.

Linux and Windows host/CUDA build scaffolding is now available. This makes loader development and validation on the
same Blackwell machine possible from either operating system; it does not move Windows production inference ahead
of the correctness and native-kernel gates below.

## Active gate

- Extend the now-committed trusted vLLM token/top-logprob fixture with full-vocabulary logits and selected hidden
  states.
- Finish the quality and native-dispatch gates for the patched same-source closest-parity GGUF, then select and lock
  quality-acceptable GGUFs for llama.cpp tiers B and C. Unpatched upstream conversion remains blocked.
- Turn the verified text-only inventory into deterministic device-arena and context-profile plans after baseline
  setup is reproducible.
- Use the retained direct vLLM characterization as a native-format performance reference. It is 1.66x–2.34x ahead
  of the patched llama.cpp candidate in prefill and 1.25x–1.26x ahead in decode through 8K, but BF16 KV capacity,
  timing-boundary differences, and autotuning fallbacks keep it from being an accepted parity baseline.

## Baseline gate

The llama.cpp benchmark is deliberately before engine kernel optimization, but after source-checkpoint validation:

1. The physical C++/Python manifest comparison must be exact. This is complete for all 1,389 tensors.
2. A trusted direct-load runtime must produce fixed token IDs and reference logits. Batch-one greedy token IDs and
   top-20 log probabilities are now committed and reproduce exactly; full-vocabulary logits remain pending.
3. The converter must emit a tensor mapping report for the same source revision. Current unpatched upstream rejects
   the checkpoint's mixed FP8/NVFP4 groups. The tracked patch now produces a 955-tensor closest-parity GGUF with 144
   NVFP4 MLP tensors and BF16-mapped attention; its exact inventory and checksum are committed.
4. Only after native SM120 NVFP4 execution, GPU residency, and quality are proven may timings be labeled as a
   closest-parity or native-NVFP4 baseline.
5. Maintain a separate fastest-practical llama.cpp baseline even if exact mixed-format parity is impossible.

## Next milestones

1. Accept or reject the patched closest-parity characterization, then establish viable llama.cpp tiers B and C.
2. Capture full-vocabulary reference logits and selected hidden states.
3. Implement the memory planner and CPU reference operators with strict golden fixtures.
4. Implement an unfused correctness engine.
5. Add and disassemble native SM120a NVFP4 kernels.
6. Specialize attention, KV cache, decode fusion, and CUDA Graph replay.
7. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.
8. After the Blackwell backend is correct and competitive, add architecture-specific backends for additional 16 GB
   CUDA GPUs without weakening benchmark or memory contracts.

The detailed gates and ordering in `AGENTS.md` are authoritative.
