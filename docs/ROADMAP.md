# Roadmap

Current stage: Milestone 2, exact numeric contracts and execution planning.

Linux and Windows host/CUDA build scaffolding is now available. This makes loader development and validation on the
same Blackwell machine possible from either operating system; it does not move Windows production inference ahead
of the correctness and native-kernel gates below.

## Active gate

- Extend the now-committed trusted vLLM token/top-logprob fixture with full-vocabulary logits and selected hidden
  states.
- Finish the quality and native-dispatch gates for the patched same-source closest-parity GGUF, then select and lock
  quality-acceptable GGUFs for llama.cpp tiers B and C. Unpatched upstream conversion remains blocked.
- Extend the implemented deterministic weight/scale/KV base arena with execution-derived activation, logits,
  sampling, graph, kernel, and prefill workspace requirements.
- Establish the exact compressed-tensors NVFP4 contract before writing a production kernel: E2M1 nibble values,
  E4M3FN local-scale decode and rounding, source nibble order, stored global-scale divisor semantics, dynamic local
  activation quantization, and FP32 accumulation. The CPU oracle is the authority for every later CUDA route.
- Prove direct source-layout consumption for the real Gate/Up `[15360,3840]` and Down `[3840,15360]` shapes.
  Each SM120 lane fragment must map to checked source byte ranges without changing an E2M1 nibble or E4M3 scale.
  A streamed final-layout transformation is permitted only if the direct route is measured and rejected; it may
  retain no raw or alternative persistent device copy.
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
3. Implement the exact host NVFP4 codec and projection oracle, including real-checkpoint byte-pattern fixtures.
4. Implement an explicit correctness-only CUDA W4A4 route that consumes packed E2M1 values and E4M3 scales. It
   must never be selected silently or reported as the native performance path.
5. Implement and round-trip-test direct SM120 fragment views over source weight/scale storage. Gate and Up may share
   one activation quantization because their global scales are identical in every pinned-checkpoint layer. Add a
   streamed final-layout transformation only if direct loads lose a measured kernel comparison.
6. Implement the native SM120a `m16n8k64` decode projection for `T=1`; disassemble it and compare it against a
   bandwidth-oriented packed-NVFP4 SIMT/GEMV candidate instead of assuming MMA wins.
7. Fuse Gate/Up with Gemma's GELU-tanh product, then implement Down plus residual. Qualify every route against the
   same CPU oracle and layer golden data.
8. Add a separate native prefill plan, initially qualified against pinned CUTLASS/cuBLASLt block-scaled GEMM. Do
   not reuse the decode plan merely for implementation convenience.
9. Implement the checkpoint's distinct FP8 attention-projection path, then assemble the unfused correctness engine.
10. Complete execution-workspace planning and add specialized attention, KV cache, decode fusion, and CUDA Graph
   replay only after the unfused model passes layer, logit, and generation gates.
11. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.
12. After the Blackwell backend is correct and competitive, add architecture-specific backends for additional 16 GB
   CUDA GPUs without weakening benchmark or memory contracts.

The detailed gates and ordering in `AGENTS.md` are authoritative.
