# Third-party dependencies

The initial runtime has no vendored third-party dependencies. It uses the C++ standard library, POSIX file mapping,
and optionally the pinned local CUDA toolkit.

Build-script structure was adapted from the neighboring `qwen35x` repository (MIT License, copyright 2026 qwen35x
contributors), inspected locally on 2026-07-21. No qwen35x runtime or loader source was copied.

The llama.cpp baseline fetches `ggml-org/llama.cpp` into the ignored `third_party/cache/llama.cpp` directory and
requires the exact commit recorded in `benchmarks/baselines/llama_cpp/commit.txt`. llama.cpp is MIT licensed. It is
used because it is the project's primary local-inference competitor and provides the comparison CUDA runtime and
HF-to-GGUF converter. No local modifications are allowed by the baseline build script. Update the commit only with
fresh conversion, correctness, instruction-path, residency, and benchmark evidence.
