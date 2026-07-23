# Memory

The deterministic base-arena planner is implemented. The verified checkpoint contains 9,304,786,336 tensor payload bytes. The explicit
text-only selection retains 9,200,026,528 bytes and skips 104,759,808 bytes of audio/vision projection and
embedding tensors. The planner separates 8,668,020,512 bytes of weights/model state from 532,006,016 bytes of
scales and aligns every named region to 256 bytes. These remain planned payload bytes, not measured CUDA allocations.

For the parsed 48-layer architecture and one-byte FP8 cache, the formula after the local window is full is:

```text
local shared K/V = 40 * min(tokens, 1024) * 8 * 256
global shared K/V = 8 * tokens * 1 * 512
separate K and V  = 2 * shared K/V
```

At 64K, this is 336 MiB for proven shared K/V semantics or 672 MiB if separate K and V storage is required. These
are formulas, not allocator measurements. Metadata, scale storage, alignment, CUDA context, workspaces, and graph
pools must be added to future measured reports.

The one-byte FP8 payload plans are:

| Profile | Context | Shared K/V | Separate K/V | Known arena, shared | Known arena, separate |
|---|---:|---:|---:|---:|---:|
| `interactive` | 8,192 | 112 MiB | 224 MiB | 8,885.83 MiB | 8,997.83 MiB |
| `standard` | 32,768 | 208 MiB | 416 MiB | 8,981.83 MiB | 9,189.83 MiB |
| `long` | 65,536 | 336 MiB | 672 MiB | 9,109.83 MiB | 9,445.83 MiB |
| `xlong` | 131,072 | 592 MiB | 1,184 MiB | 9,365.83 MiB | 9,957.83 MiB |
| `max` | 262,144 | 1,104 MiB | 2,208 MiB | 9,877.83 MiB | 10,981.83 MiB |

Every plan reports both cache possibilities and requires the selected layout explicitly. Checked multiplication,
addition, and alignment reject integer overflow. Activation A/B, logits, sampling, CUDA Graph, kernel, and prefill
workspaces remain explicitly unplanned until their execution shapes are defined; `total_arena_bytes` is therefore
the known base arena, not a peak-VRAM claim.
