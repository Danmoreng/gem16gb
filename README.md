# gem16gb

`gem16gb` is an early-stage C++20/CUDA inference engine for high-performance inference on NVIDIA GPUs with
approximately 16 GB of VRAM. The first model is the mixed FP8/NVFP4
`unsloth/gemma-4-12b-it-NVFP4` checkpoint, and the first optimized backend is Blackwell SM120/SM120a.

## What works

- The model repository is pinned to commit `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, including file sizes,
  SHA-256 digests, Git object IDs, and Xet/LFS identities.
- `gem16gb-inspect` memory-maps a single Safetensors file or indexed shards, validates offsets and byte lengths,
  parses the compressed-tensors quantization schema, classifies all 1,389 tensors in the pinned snapshot, and
  exports JSON.
- Host parser tests work on Linux and Windows without CUDA. Linux also has opt-in ASan/UBSan builds.
- `gem16gb-bench memory` builds an aligned, deterministic base arena from the real text-only tensor inventory and
  reports both shared and separate K/V cache sizes for every context profile.
- The exact host NVFP4 codec covers E2M1, E4M3FN, dynamic-local activation quantization, compressed-tensors global
  divisors, and a binary64 projection oracle with pinned-checkpoint byte fixtures.
- The CUDA build contains an explicit correctness-only W4A4 projection and an experimental direct-source SM120a
  projection. Synthetic tests prove CUDA intrinsic agreement and native block-scaled MMA output, but no complete
  layer or model path is qualified yet.

Inference and benchmarks do **not** work yet. `gem16gb-run` and `gem16gb-bench` fail visibly and never fall back to a
higher precision path.

## Build on Linux

CMake 3.28+, Ninja, and a C++20 compiler are required.

```bash
./scripts/build.sh --host --test
```

For the CUDA capability probe with the pinned local toolkit:

```bash
./scripts/build.sh --cuda --test
build/Linux/blackwell-release/bin/gem16gb-run --print-kernel-capabilities
```

The CUDA build targets only `120a`. Its experimental projection disassembles to
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, but the capability report deliberately remains
`native_nvfp4_kernels=false` until real-shape, layer, logit, and benchmark gates pass.

## Build on Windows

Use PowerShell from a regular terminal; the helper discovers Visual Studio 2022 Build Tools with `vswhere`, imports
the x64 MSVC environment, and uses Ninja. CMake 3.28+, Ninja, and the Visual Studio C++ workload are required.

```powershell
.\scripts\build.ps1 -Test
```

For the CUDA capability probe, install the pinned CUDA toolkit. The helper uses `CUDA_PATH` when set and otherwise
discovers toolkits installed in NVIDIA's standard Windows location:

```powershell
.\scripts\build.ps1 -Cuda -Test
.\build\Windows\blackwell-release\bin\gem16gb-run.exe --print-kernel-capabilities
```

The target layout is the same on both operating systems, while CMake caches stay isolated under `build/Linux` and
`build/Windows`. The `host-sanitize` preset is Linux-only because it currently uses GCC/Clang ASan and UBSan.
The validated Windows development toolchain is recorded in `toolchains/windows-blackwell16gb.lock`; the Linux
reference remains in `toolchains/blackwell16gb.lock`.

## Download the pinned checkpoint

The checkpoint is about 9.34 GB. `HF_TOKEN` is optional for this public repository.

```bash
python3 tools/fetch_model.py \
  --destination models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

On Windows, use `python` instead of `python3`; all Python tools use `pathlib` and accept native Windows paths.
The trusted vLLM reference runtime remains Linux-only because upstream vLLM has no supported native Windows
runtime; run those reference-generation and characterization commands on Linux rather than changing their
semantics.

The downloader resumes partial files, verifies sizes and SHA-256 digests, and never imports or executes code
from the model repository.

## Inspect

```bash
build/Linux/host-debug/bin/gem16gb-inspect \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --validate \
  --json manifest.json
```

The equivalent Windows command is:

```powershell
.\build\Windows\host-debug\bin\gem16gb-inspect.exe `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --validate `
  --json .\manifest.json
```

## Plan memory

K/V sharing is not assumed silently. Select the intended storage explicitly; the JSON result also contains the
alternative size:

```powershell
.\build\Windows\host-debug\bin\gem16gb-bench.exe memory `
  --model .\models\checkpoints\unsloth-gemma-4-12b-it-NVFP4-b1f6497 `
  --profile long `
  --kv-storage shared
```

The current base plan covers immutable text weights, scales, and KV payload. Activation, graph, sampling, kernel,
and prefill workspaces remain explicitly marked as unplanned rather than being estimated without an execution plan.

## Hardware and limitations

Blackwell is the first implementation target, not a permanent board-specific product boundary. Every benchmark
records the exact board, power, clocks, driver, and VRAM independently, while engine planning is expressed in terms
of the 16 GB CUDA hardware class. No engine performance claim exists yet.

Project code is Apache-2.0. Checkpoint use is governed separately by its model card and linked Gemma terms.
