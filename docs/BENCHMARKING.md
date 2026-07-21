# Benchmarking

There are no accepted comparative benchmark results yet. `gem16gb-bench` returns a machine-readable
`not_implemented` status and a non-zero exit code.

Future results must use the matrices, timing boundaries, repetition policy, quality gates, and three llama.cpp
baseline labels defined in `AGENTS.md`. Raw runs will be written below
`benchmarks/results/<date>/<git-sha>/<machine-id>/` and never overwritten. Throughput speedup and latency reduction
must be reported separately. Exact board identity, power envelope, clocks, and thermals are recorded for every run;
the project scope remains the 16 GB CUDA target class.

Current upstream llama.cpp is pinned, but its unpatched converter rejects the locked checkpoint's mixed FP8/NVFP4
compressed-tensors groups. A tracked converter patch produces a closest-parity candidate that preserves NVFP4 MLP
tensors and maps FP8 attention weights to BF16. Its inventory, direct-runtime quality comparison, and full-residency
probe are recorded; native-path profiling and quality acceptance remain open gates. This candidate cannot be labeled
exact format parity.

The patched llama.cpp candidate has a retained development characterization covering prefill through 65,536
tokens and decode at context depths through 8,192. Its tracked summary is
`benchmarks/baselines/llama_cpp/characterization.json`; raw samples are retained under `benchmarks/results/`. It is
not an accepted baseline because native dispatch profiling, a quality threshold, inter-token latency capture, and
power/clock/thermal telemetry remain open.

The pinned SM120a competitor build is complete. Its NVFP4 object contains native block-scaled
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions. This is a binary capability check only; runtime dispatch must
still be captured with the selected model before a tier-B result is valid.
