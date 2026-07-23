# Checkpoint format

## Pinned source

- Repository: `unsloth/gemma-4-12b-it-NVFP4`
- Revision: `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`
- Compressed-tensors version declared by config: `0.17.2.a20260707`
- Lock: `models/gemma4-12b-nvfp4.lock.json`

This is a mixed checkpoint, not an all-NVFP4 checkpoint. The pinned config targets attention projections with
per-channel FP8 weights and per-token dynamic FP8 inputs. It targets language-model MLP gate/up/down projections
with packed NVFP4 weights, group size 16, E4M3 local scales, and tensor-global scales.

The verified source contains 1,389 tensors. NVFP4 modules use `.weight_packed`, `.weight_scale`,
`.weight_global_scale`, and `.input_global_scale`. Packed U8 shapes halve the logical contracting dimension;
local scales are E4M3 with one value per 16 logical contracting elements, and both global scales are scalar F32.
Attention projection weights are E4M3 with BF16 per-output-channel `.weight_scale` tensors. Sliding-attention
layers contain named `v_proj` tensors; the eight full-attention layers omit `v_proj` weight and scale tensors under
unified K/V semantics. Execution must follow this per-layer inventory. Nibble order still requires byte-pattern
validation before kernel work. No persistent repacked layout is defined.

Compressed-tensors stores the NVFP4 global values as divisors. For a stored weight divisor `gw`, input divisor
`ga`, packed E2M1 values `qw`, and local E4M3FN scales `sw`, the expected W4A4 execution contract is:

```text
w_real       = qw * sw / gw
a_scaled     = a_real * ga
a_scaled     ~= qa * sa                 # dynamic E2M1 plus E4M3FN scale per 16 values
projection   = sum(qa * sa * qw * sw) / (ga * gw)
```

This interpretation must be verified against the pinned trusted runtime and CPU oracle before it becomes a kernel
contract. Gate and Up divisors are bit-identical for all 48 layers in the pinned checkpoint. All 530,841,600 bytes
of the 144 MLP local-scale tensors are positive, nonzero, and avoid the E4M3FN NaN encoding.
