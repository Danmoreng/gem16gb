# Roadmap

Current stage: Milestone 3, native layer assembly.

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
- Keep the implemented CPU NVFP4 oracle as the authority while assembling the first complete MLP layer. The exact
  E2M1, E4M3FN, source-nibble, global-divisor, activation-quantization, and FP32-accumulation contracts are now
  executable and tested.
- Assemble Gate/Up, GELU-tanh product, Down, and residual without weakening the now-complete real-checkpoint proof:
  all three Layer-0 shapes consume source weight and scale storage directly, CPU/GPU activation bytes match exactly,
  and CUDA reference/native output differences are at most `1.1920929e-7` in the characterization fixture.
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
3. ~~Implement the exact host NVFP4 codec and projection oracle, including real-checkpoint byte-pattern fixtures.~~
4. ~~Implement an explicit correctness-only CUDA W4A4 route that consumes packed E2M1 values and E4M3 scales.~~
5. ~~Implement and round-trip-test direct SM120 fragment views over source weight/scale storage for Gate, Up, and
   Down.~~ No persistent repack is required by the current direct route.
6. Continue tuning the implemented SM120a `m16n8k64` decode projection for `T=1`; disassembly already proves
   `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. Compare it against a bandwidth-oriented packed-NVFP4 SIMT/GEMV candidate
   before declaring the production winner.
7. The correctness-first Layer-0 MLP chain now implements Gate/Up, Gemma GELU-tanh product, Down, and residual with
   two exact CPU/CUDA NVFP4 quantization boundaries. Next compare it with trusted layer golden data, then fuse
   launches and remeasure without using the isolated hot-cache numbers as a layer estimate.
8. Add a separate native prefill plan, initially qualified against pinned CUTLASS/cuBLASLt block-scaled GEMM. Do
   not reuse the decode plan merely for implementation convenience.
9. The checkpoint's FP8 Q/K/V/O projection path is now implemented with an independent CPU oracle, CUDA reference,
   direct-source `QMMA.16832` route, and real Layer-0 checks. Next assemble RMSNorm, Q/K normalization, RoPE,
   local/global attention, KV append/read, O projection, and residual into the unfused correctness engine. Follow
   ninfer's useful split-output planning pattern for combined Q/K/V, while retaining this checkpoint's E4M3/BF16
   scale contract and distinct full-attention K=V inventory.
10. Complete execution-workspace planning and add specialized attention, KV cache, decode fusion, and CUDA Graph
   replay only after the unfused model passes layer, logit, and generation gates.
11. Validate 64K, then 128K context. MTP and multimodal work remain later milestones.
12. After the Blackwell backend is correct and competitive, add architecture-specific backends for additional 16 GB
   CUDA GPUs without weakening benchmark or memory contracts.

The detailed gates and ordering in `AGENTS.md` are authoritative.
