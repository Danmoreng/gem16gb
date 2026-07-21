# Benchmarking

There are no valid benchmark results yet. `g4-bench` returns a machine-readable `not_implemented` status and a
non-zero exit code.

Future results must use the matrices, timing boundaries, repetition policy, quality gates, and three llama.cpp
baseline labels defined in `AGENTS.md`. Raw runs will be written below
`benchmarks/results/<date>/<git-sha>/<machine-id>/` and never overwritten. Throughput speedup and latency reduction
must be reported separately; laptop-GPU measurements are development diagnostics only.

Current upstream llama.cpp is pinned, but its converter rejects the locked checkpoint's mixed FP8/NVFP4
compressed-tensors groups. Consequently there is no tier-A GGUF and no valid same-source timing yet. The baseline
order is conversion/tensor inventory, direct-runtime quality comparison, full GPU residency and native-path proof,
then timing. Tiers B and C may proceed with separately identified GGUFs, but they cannot be labeled format parity.

The pinned SM120a competitor build is complete. Its NVFP4 object contains native block-scaled
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions. This is a binary capability check only; runtime dispatch must
still be captured with the selected model before a tier-B result is valid.
