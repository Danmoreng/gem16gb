# Decisions

## 2026-07-23: Bring up NVFP4 from an exact oracle into separate decode and prefill plans

Date: 2026-07-23
Decision: Implement the E2M1/E4M3FN and compressed-tensors divisor contract first, followed by an explicit
correctness CUDA route, direct source-layout SM120 fragment views, and independently measured packed-GEMV and native-MMA decode
candidates. Fuse Gate/Up only after the common input/global-scale invariant is validated; build prefill as a
separate plan and keep FP8 attention as a separate precision backend.
Context: NInfer demonstrates the value of closed, shape-specific plan catalogs, arenas, graph-stable addresses, and
Gate/Up fusion, but its integer Q4 format and offline `.ninfer` artifact are incompatible with this checkpoint.
A neighboring SM120 prototype demonstrates the native block-scaled instruction and operand-fragment mapping, but it
retains multiple device layouts and previously exposed an input-global-scale semantic error. The pinned Gemma
checkpoint stores compressed-tensors global divisors, has exact SM120-friendly MLP dimensions, and uses mixed FP8
attention plus NVFP4 MLP projections.
Alternatives: Start with a complete unfused model and debug quantization indirectly; copy the neighboring loader and
retain raw plus multiple repacked device tensors; assume native MMA is fastest at `T=1`; use one GEMM plan for decode
and prefill.
Consequences: Kernel work begins later but every route shares one independent oracle. The 16 GB memory contract is
preserved, silent precision fallback remains impossible, and the project obtains direct evidence for the actual
batch-one winner. Gate/Up can reuse one activation quantization and later fuse the GELU-tanh epilogue. The first
native candidate adds no persistent repacked weight or expanded scale copy; a streamed transformation remains a
measured fallback. Loader and kernel layouts remain architecture-specific implementation details behind the
manifest contract.
Evidence: All 48 Gate/Up pairs have identical stored input and weight divisors. The real Gate/Up and Down shapes are
divisible by the intended 128/64 outer/contracting geometry. The 144 local-scale tensors contain 530,841,600
positive, nonzero E4M3FN bytes with no NaN encoding.

## 2026-07-23: Keep the first memory plan explicit and evidence-bounded

Date: 2026-07-23
Decision: Build a deterministic 256-byte-aligned base arena from the parsed text-only tensor inventory and context
metadata. Calculate both shared and separate K/V payloads, require an explicit selection, and leave execution
workspaces visibly unplanned until kernel shapes define them.
Context: The checkpoint proves `attention_k_eq_v=true`, but physical shared-cache semantics and workspace sizes have
not yet been validated by an executable model path. A 16 GB budget cannot tolerate hidden or guessed allocations.
Alternatives: Assume shared K/V immediately; reserve budget-table maxima as real allocations; defer all memory work
until CUDA kernels exist.
Consequences: Weight, scale, and KV offsets are deterministic and overflow-checked now. Memory reports remain useful
without claiming peak VRAM. The plan is deliberately incomplete until activations, logits, sampling, graph, kernel,
and prefill workspaces are derived and measured.
Evidence: The locked 1,389-tensor manifest yields 9,200,026,528 text-only bytes. At 64K, parsed layer metadata yields
336 MiB shared or 672 MiB separate one-byte K/V payloads, matching the independently documented formula.

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
