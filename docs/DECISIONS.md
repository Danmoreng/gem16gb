# Decisions

## 2026-07-21: Keep CUDA opt-in during repository initialization

Date: 2026-07-21  
Decision: Provide separate host-debug and RTX-5080 CUDA presets. Do not label the CUDA runtime probe as a native
kernel path.  
Context: Parser and manifest work must build on machines without CUDA, while performance builds must remain
architecture-specific.  
Alternatives: Require CUDA for every build; silently build host-only when CUDA is absent.  
Consequences: CPU CI stays useful; `G4_ENABLE_CUDA=ON` fails if CUDA is missing; native capability remains false
until implemented.  
Evidence: The neighboring `qwen35x` repository successfully uses optional CUDA language enablement, but its
silent CPU fallback was tightened here to a fatal error when CUDA is explicitly requested.

## 2026-07-21: Implement a strict in-repository JSON parser

Date: 2026-07-21  
Decision: Use a small C++ parser with duplicate-key rejection, resource limits, Unicode validation, and checked
integer parsing for initial config and Safetensors work.  
Context: Runtime dependency count should remain small and model files are untrusted input.  
Alternatives: Vendor a JSON library immediately; use string searching.  
Consequences: The parser is narrowly testable and dependency-free, but it carries maintenance responsibility and
must be fuzzed before the loader is considered production-ready.  
Evidence: Neighboring ad-hoc string-search Safetensors code does not meet this repository's schema and security
requirements.

## 2026-07-21: Do not use laptop-GPU results as primary RTX 5080 results

Date: 2026-07-21  
Decision: Treat the detected RTX 5080 Laptop GPU as a development device only.  
Context: The project contract targets a desktop RTX 5080 and fixed hardware conditions.  
Alternatives: Treat all devices with compute capability 12.0 as benchmark-equivalent.  
Consequences: Correctness and development can proceed locally, but primary performance claims require the exact
target machine.  
Evidence: `nvidia-smi` identifies this machine as `NVIDIA GeForce RTX 5080 Laptop GPU` with 16,303 MiB.

