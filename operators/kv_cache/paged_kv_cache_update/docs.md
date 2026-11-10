# paged_kv_cache_update

## Function

Writes new token states into flattened slots of a paged KV cache.

## Inputs

- `to_cache`: token state tensor with shape `[num_tokens, num_heads, head_dim]`.
- `kv_cache`: paged cache tensor with shape `[num_blocks, block_size, num_heads, head_dim]`.
- `slot_mapping`: cache slot index tensor with shape `[num_tokens]`.

`to_cache` and `kv_cache` use the dtype selected by the shape entry. `slot_mapping` uses `index_dtype`.

## Output

Returns one updated paged cache tensor with shape `[num_blocks, block_size, num_heads, head_dim]`.

## Implementation Notes

- Treat `kv_cache` as flattened slots with shape `[num_blocks * block_size, num_heads, head_dim]`.
- For each token `i`, if `slot_mapping[i] >= 0`, copy `to_cache[i]` into the flattened cache slot `slot_mapping[i]`.
- If `slot_mapping[i] < 0`, skip that token and leave the cache unchanged.
- Preserve cache slots that are not listed in `slot_mapping`.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache write/copy operator; preserve source values exactly in the configured dtype.
- Do not rescale, normalize, or convert cache values through FP32 for pure writes.
- Keep slot/index handling exact and preserve cache entries that are not written.
