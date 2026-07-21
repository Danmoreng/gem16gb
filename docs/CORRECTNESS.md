# Correctness

## Implemented level

Level 0 currently covers bounded JSON parsing, duplicate-key rejection, little-endian Safetensors header lengths,
shape-product overflow, known dtype byte sizes, payload bounds, exact byte lengths, overlapping ranges, duplicate
tensors across shards, index agreement, UTF-8 strings, shard path traversal rejection, and symlink escape rejection.

`g4-inspect --validate` additionally checks the expected primary architecture dimensions and quantization mode,
then requires each classified NVFP4 packed weight to have local, global, and input scale tensors.

## Not yet established

Operator tolerances, reference logits, generation agreement, and task quality have not been measured. Therefore
`tests/tolerances.yaml` is intentionally empty. Tolerances will be added only after reference distributions exist.
