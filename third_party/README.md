# Third-party dependencies

The initial runtime has no vendored third-party dependencies. It uses the C++ standard library, POSIX file mapping,
and optionally the pinned local CUDA toolkit.

Build-script structure was adapted from the neighboring `qwen35x` repository (MIT License, copyright 2026 qwen35x
contributors), inspected locally on 2026-07-21. No qwen35x runtime or loader source was copied.

The llama.cpp baseline fetches `ggml-org/llama.cpp` into the ignored `third_party/cache/llama.cpp` directory and
requires the exact commit recorded in `benchmarks/baselines/llama_cpp/commit.txt`. llama.cpp is MIT licensed. It is
used because it is the project's primary local-inference competitor and provides the comparison CUDA runtime and
HF-to-GGUF converter. The upstream baseline build remains unmodified. A separately labeled, tracked converter patch
is applied only to an ignored worktree by `prepare-patched-source.sh`; its purpose, exact mapping, and SHA-256 are
recorded under `benchmarks/baselines/llama_cpp/`. Update the commit or patch only with fresh conversion,
correctness, instruction-path, residency, and benchmark evidence.
