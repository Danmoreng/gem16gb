# Memory

No GPU arena exists yet. The verified checkpoint contains 9,304,786,336 tensor payload bytes. The explicit
text-only selection retains 9,200,026,528 bytes and skips 104,759,808 bytes of audio/vision projection and
embedding tensors. These are source payload measurements, not GPU allocation measurements.

For the parsed 48-layer architecture and one-byte FP8 cache, the formula after the local window is full is:

```text
local shared K/V = 40 * min(tokens, 1024) * 8 * 256
global shared K/V = 8 * tokens * 1 * 512
separate K and V  = 2 * shared K/V
```

At 64K, this is 336 MiB for proven shared K/V semantics or 672 MiB if separate K and V storage is required. These
are formulas, not allocator measurements. Metadata, scale storage, alignment, CUDA context, workspaces, and graph
pools must be added to future measured reports.
