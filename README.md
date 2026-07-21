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
- Host parser tests and opt-in ASan/UBSan builds work without CUDA.
- An SM120a CUDA capability build is wired, but contains no inference kernels yet.

Inference and benchmarks do **not** work yet. `gem16gb-run` and `gem16gb-bench` fail visibly and never fall back to a
higher precision path.

## Build

```bash
./scripts/build.sh --host --test
```

For the CUDA capability probe with the pinned local toolkit:

```bash
./scripts/build.sh --cuda --test
build/blackwell-release/bin/gem16gb-run --print-kernel-capabilities
```

The CUDA build targets only `120a`. It does not claim native NVFP4 support until a representative kernel is
implemented and its instructions are verified.

## Download the pinned checkpoint

The checkpoint is about 9.34 GB. `HF_TOKEN` is optional for this public repository.

```bash
python3 tools/fetch_model.py \
  --destination models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497
```

The downloader resumes partial files, verifies sizes and SHA-256 digests, and never imports or executes code
from the model repository.

## Inspect

```bash
build/host-debug/bin/gem16gb-inspect \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --validate \
  --json manifest.json
```

## Hardware and limitations

Blackwell is the first implementation target, not a permanent board-specific product boundary. Every benchmark
records the exact board, power, clocks, driver, and VRAM independently, while engine planning is expressed in terms
of the 16 GB CUDA hardware class. No engine performance claim exists yet.

Project code is Apache-2.0. Checkpoint use is governed separately by its model card and linked Gemma terms.
