# Correctness

## Implemented level

Level 0 currently covers bounded JSON parsing, duplicate-key rejection, little-endian Safetensors header lengths,
shape-product overflow, known dtype byte sizes, payload bounds, exact byte lengths, overlapping ranges, duplicate
tensors across shards, index agreement, UTF-8 strings, shard path traversal rejection, and symlink escape rejection.

`g4-inspect --validate` additionally checks the expected primary architecture dimensions and quantization mode,
then requires each classified NVFP4 packed weight to have local, global, and input scale tensors.

An independent Python reader compares the raw Safetensors headers against the exported C++ manifest. For pinned
revision `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`, all 1,389 tensors match across physical shape, dtype, absolute
offset, byte length, shard, and alignment; total tensor payload is 9,304,786,336 bytes with zero mismatches. The
validated decoder inventory contains 29 tensors in every sliding-attention layer and 27 in every full-attention
layer; full-attention layers omit separate `v_proj` weight and scale tensors.

## Not yet established

Operator tolerances, reference logits, generation agreement, and task quality have not been measured. Therefore
`tests/tolerances.yaml` is intentionally empty. Tolerances will be added only after reference distributions exist.

Reproduce the physical manifest comparison with:

```bash
build/host-debug/bin/g4-inspect --model <checkpoint> --validate --json build/manifest.json
python3 tools/compare_manifests.py --model <checkpoint> --manifest build/manifest.json
```
