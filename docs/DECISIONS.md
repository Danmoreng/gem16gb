# Decisions

## 2026-07-23: Support Linux and Windows in the repository foundation

Date: 2026-07-23
Decision: Keep one Ninja-based preset layout for both operating systems, isolate file mapping behind POSIX and Win32
implementations, and provide native Bash and PowerShell build entry points. Add Windows host CI while retaining the
Linux sanitizer path.
Context: Development moved from Linux to Windows on the same Blackwell machine. Loader and build work must remain
reproducible on both systems without weakening the Linux reference path.
Alternatives: Develop only through WSL; maintain unrelated Windows CMake targets; replace memory mapping with full
file reads.
Consequences: Host and SM120a capability builds share target names and their internal `bin`/`lib` layout, while
OS-named build roots prevent incompatible CMake caches from colliding. Windows uses Unicode-aware Win32 file
mapping and self-discovers MSVC through Visual Studio Build Tools. ASan/UBSan remains Linux-only until a
Windows sanitizer configuration provides comparable signal. Linux remains the production platform required by the
phase-one contract, while Windows is now a supported development and validation host.
Evidence: On the reference Windows installation, MSVC 19.44 and CUDA 13.3 configure and build both presets with
warnings as errors; host and CUDA CTest runs pass.

## 2026-07-21: Keep CUDA opt-in during repository initialization

Date: 2026-07-21  
Decision: Provide separate host-debug and Blackwell CUDA presets. Do not label the CUDA runtime probe as a native
kernel path.  
Context: Parser and manifest work must build on machines without CUDA, while performance builds must remain
architecture-specific.  
Alternatives: Require CUDA for every build; silently build host-only when CUDA is absent.  
Consequences: CPU CI stays useful; `GEM16GB_ENABLE_CUDA=ON` fails if CUDA is missing; native capability remains false
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

## 2026-07-21: Target the 16 GB CUDA hardware class, Blackwell first

Date: 2026-07-21  
Decision: Define the product target as NVIDIA CUDA GPUs with approximately 16 GB VRAM. Optimize and validate the
first backend on the available Blackwell compute-capability-12.0 GPU.
Context: The engine should become useful across the 16 GB CUDA class; retail board form factors do not belong in the
architecture or project identity.
Alternatives: Bind the project to one retail board; attempt multi-architecture kernels before the first backend is
correct and competitive.
Consequences: Blackwell remains the immediate kernel and benchmark target. Later GPU backends must preserve the same
correctness, memory, and benchmark contracts, and exact board details remain benchmark metadata rather than product
scope.
