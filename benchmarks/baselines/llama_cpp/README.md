# llama.cpp baseline

Upstream is pinned to commit `846e991ec3c7ccec49112ff2c5b00b710e5f551d`, the `master` tip resolved on
2026-07-21. No performance baseline is established yet.

## Same-source gate

The exact pinned Hugging Face checkpoint cannot currently pass upstream's converter. The converter recognizes
`Gemma4UnifiedForConditionalGeneration`, indexes the Safetensors file, and then rejects its two compressed-tensors
configuration groups:

```text
NotImplementedError: Can't handle multiple config groups for compressed-tensors yet
```

This is expected from the source: its mixed-precision shortcut only accepts multiple groups when every group is
`nvfp4-pack-quantized`; this checkpoint combines FP8 attention and NVFP4 MLP groups. See
`conversion-probe.json` for the exact command and result. This blocks tier A conversion and timing; it does not
justify changing or flattening tensor precision without a manifest and quality review.

Reproduce the converter gate after preparing its documented Python requirements:

```bash
benchmarks/baselines/llama_cpp/convert.sh \
  models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  build/llama_cpp/gemma4-12b-nvfp4.gguf \
  --dry-run
```

`build.sh` checks out the exact clean commit and builds CUDA tools specifically for SM120a. The build alone does
not establish native NVFP4 execution. Before `run.sh` is enabled, the selected GGUF still needs a tensor inventory,
full GPU-residency proof, native-instruction/path evidence, and quality comparison against the direct checkpoint.

The current build's dedicated `mmq-instance-nvfp4.cu.o` is an `sm_120a` cubin. `cuobjdump` confirms that it contains
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, matching the block-scaled E2M1/UE4M3 native path. This proves instruction
availability in the binary, not that a particular model invokes it.

The three required tiers remain separate:

1. same-source closest parity (currently blocked at conversion);
2. native-NVFP4 llama.cpp (model selection and quality gate pending);
3. fastest practical quality-acceptable llama.cpp (model selection and quality gate pending).
