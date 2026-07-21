# llama.cpp baseline

Upstream is pinned to commit `846e991ec3c7ccec49112ff2c5b00b710e5f551d`, the `master` tip resolved on
2026-07-21. A development characterization has been captured; no accepted or headline baseline exists yet.

## Same-source gate

The exact pinned Hugging Face checkpoint cannot currently pass upstream's converter. The converter recognizes
`Gemma4UnifiedForConditionalGeneration`, indexes the Safetensors file, and then rejects its two compressed-tensors
configuration groups:

```text
NotImplementedError: Can't handle multiple config groups for compressed-tensors yet
```

This is expected from the source: its mixed-precision shortcut only accepts multiple groups when every group is
`nvfp4-pack-quantized`; this checkpoint combines FP8 attention and NVFP4 MLP groups. See
`conversion-probe.json` for the exact command and result. Upstream tier A therefore remains blocked.

An auditable local patch now enables a separately labeled **same-source closest-parity patched** candidate. It
preserves and repacks the 144 NVFP4 MLP tensors, then explicitly dequantizes the 184 FP8 attention weights to BF16.
This is not exact format parity. The generated GGUF has 955 tensors and SHA-256
`e4910e01c4275e58acbf2c38c4d4fb81acf61bb8aa04eed121eb5ac942705e8a`; see `patched-conversion.json` and
`tensor-inventory.json`.

Reproduce the converter gate after preparing its documented Python requirements:

```bash
benchmarks/baselines/llama_cpp/convert.sh \
  models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  build/llama_cpp/gemma4-12b-nvfp4.gguf \
  --dry-run
```

Prepare and run the patched converter in a separate ignored worktree:

```bash
benchmarks/baselines/llama_cpp/prepare-patched-source.sh
benchmarks/baselines/llama_cpp/convert-patched.sh \
  models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  build/llama_cpp/gemma4-12b-nvfp4-patched.gguf
```

The checkpoint tokenizer metadata requires Transformers 5 for this probe; Transformers 4.57.6 from upstream's
legacy converter requirements fails while reading `extra_special_tokens`. The default patched command therefore
uses the already pinned offline reference environment with Transformers 5.14.1. This is converter tooling only,
not a runtime dependency.

`build.sh` checks out the exact clean commit and builds CUDA tools specifically for SM120a. The build alone does
not establish native NVFP4 execution. The selected GGUF has passed structural inspection, full GPU-residency
probing, and an initial direct-runtime quality comparison; profiler-level native-instruction dispatch evidence is
still required.

The current build's dedicated `mmq-instance-nvfp4.cu.o` is an `sm_120a` cubin. `cuobjdump` confirms that it contains
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, matching the block-scaled E2M1/UE4M3 native path. This proves instruction
availability in the binary. The candidate loads 49/49 layers on the GPU and uses 11,069.25 MiB for model tensors,
608 MiB for the 8K BF16 KV/context allocation, and 115.52 MiB of compute space, leaving 3,837 MiB free. A
profiler-level trace of the native kernel invocation remains an explicit gate.

The direct-runtime comparison in `quality.json` uses identical chat rendering, greedy sampling, and the same three
prompts. It records 50/65 token agreement and one exact generation. The short sky answer diverges after 18 matching
tokens; the thinking trace matches 28/32 tokens. These are measurements, not an adopted tolerance. Timing results
remain characterization-only until a quality threshold and native dispatch trace are approved.

`characterization.json` summarizes 10 measured runs after three conditioning runs for the required prefill points
through 65,536 tokens and decode points at context depths 128, 2,048, and 8,192. It retains every throughput sample,
reports the median as primary, and includes mean, sample standard deviation, range, and a 95% Student-t confidence
interval. The raw inputs remain under `benchmarks/results/`. This run did not capture profiler dispatch,
per-token latency distributions, or power/clock/thermal time series, so it is development evidence only.

The three required tiers remain separate:

1. same-source closest parity (upstream blocked; patched candidate under characterization);
2. native-NVFP4 llama.cpp (model selection and quality gate pending);
3. fastest practical quality-acceptable llama.cpp (model selection and quality gate pending).
