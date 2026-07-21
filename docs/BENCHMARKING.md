# Benchmarking

There are no valid benchmark results yet. `g4-bench` returns a machine-readable `not_implemented` status and a
non-zero exit code.

Future results must use the matrices, timing boundaries, repetition policy, quality gates, and three llama.cpp
baseline labels defined in `AGENTS.md`. Raw runs will be written below
`benchmarks/results/<date>/<git-sha>/<machine-id>/` and never overwritten. Throughput speedup and latency reduction
must be reported separately; laptop-GPU measurements are development diagnostics only.

