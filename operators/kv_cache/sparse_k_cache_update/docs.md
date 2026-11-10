# sparse_k_cache_update

## Function

Writes sparse attention key rows into a preallocated key cache using a slot mapping.

## Inputs

- `k`: key tensor with shape `[num_tokens, head_dim]`.
- `slot_mapping`: int32 tensor with shape `[num_tokens]`; each valid entry selects one row in `kv_cache`.
- `kv_cache`: key cache tensor with shape `[num_slots, head_dim]`.

All key and cache tensors use the dtype selected by the shape entry.

## Output

- `kv_cache`: updated key cache with shape `[num_slots, head_dim]` and dtype `dtype`.

## Implementation Notes

- This is a KV cache update helper, not attention score/value aggregation.
- Copy each valid `k[token]` row into `kv_cache[slot_mapping[token]]`.
- `slot_mapping` entries are unique in the generated benchmark inputs.
- Implement the cache update in backend-native kernel code.
- Do not call PyTorch advanced indexing, scatter, index assignment, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache write/copy operator; preserve source key values exactly in the configured dtype.
- Do not rescale, normalize, or convert cache values through FP32 for pure writes.
- Keep slot mapping exact and preserve cache entries that are not written.
