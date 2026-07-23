# Performance ledger

Kernel characterizations below are development evidence, not accepted end-to-end benchmark claims. They use one
deterministic activation and the pinned Layer-0 checkpoint tensors on the current Windows Blackwell machine.
Repeated isolated projection measurements keep one 33.3 MB tensor family hot in cache; do not add their times to
estimate a layer. The complete MLP row cycles through the 99.5 MB three-projection working set and is the more useful
decode characterization.

| Date | Commit | Hypothesis | Configuration | Before | After | Quality delta | VRAM delta | Decision |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-07-23 | working tree | Direct source-layout SM120 MMA can consume the checkpoint without persistent repack | Gate `[15360,3840]`, W4A4 NVFP4, 3 warm-ups/10 iterations | CUDA scalar reference 0.2785 ms | SM120 direct 0.0334 ms | max abs `1.1920929e-7`; cosine `0.9999999999999999` | 0 persistent repack bytes | Retain direct route; continue qualification |
| 2026-07-23 | working tree | The same mapping is valid for Up | Up `[15360,3840]`, W4A4 NVFP4, 3/10 | CUDA scalar reference 0.2784 ms | SM120 direct 0.0288 ms | max abs `5.9604645e-8`; cosine `1.0` | 0 persistent repack bytes | Retain direct route |
| 2026-07-23 | working tree | The direct route also covers the transposed logical Down shape | Down `[3840,15360]`, W4A4 NVFP4, 3/10 | CUDA scalar reference 0.6604 ms | SM120 direct 0.0412 ms | max abs `0`; cosine `1.0` | 0 persistent repack bytes | Retain direct route |
| 2026-07-23 | working tree | The exact operators compose without a numerical discontinuity at Down requantization | Complete Layer-0 MLP plus residual, W4A4 NVFP4, 10 warm-ups/100 iterations | CUDA scalar-reference chain 1.2648 ms | SM120 direct chain 0.4951 ms | zero differing Down-input bytes; final max abs `0`; oracle max abs `6.7374888e-9` | 99,774,000 probe device bytes; 0 persistent repack bytes | Proceed to trusted hidden-state comparison and fusion |

