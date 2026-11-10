# concat_and_cache_mla_rope

## Function

Applies RoPE to MLA query/key positional features, concatenates the latent KV feature with the rotated key feature, and writes the fused entry into paged KV cache slots.

This is a fused MLA cache update operator, not a full attention operator.

## Inputs

- `positions`: position indices with shape `[num_tokens]`.
- `q_pe`: query positional feature tensor with shape `[num_tokens, rope_dim]`.
- `k_pe`: key positional feature tensor with shape `[num_tokens, rope_dim]`.
- `kv_c`: latent KV tensor with shape `[num_tokens, latent_dim]`.
- `cos_sin_cache`: RoPE cosine/sine cache with shape `[max_position, rope_dim]`.
- `slot_mapping`: physical cache slot indices with shape `[num_tokens]`; negative slots are ignored.
- `kv_cache`: cache tensor with shape `[num_blocks, block_size, latent_dim + rope_dim]`.

`q_pe`, `k_pe`, `kv_c`, and `kv_cache` use the dtype selected by the shape entry. `cos_sin_cache` is FP32. `positions` and `slot_mapping` are INT64.

## Output

Returns a tuple:

- `q_rot`: rotated query positional feature tensor with shape `[num_tokens, rope_dim]`.
- `k_rot`: rotated key positional feature tensor with shape `[num_tokens, rope_dim]`.
- `cached`: updated KV cache tensor with shape `[num_blocks, block_size, latent_dim + rope_dim]`.

Floating outputs use the dtype selected by the shape entry.

## Implementation Notes

- Implement RoPE rotation, latent/key concatenation, slot filtering, and cache write in backend-native kernel code.
- Support both NeoX split-half RoPE layout and interleaved RoPE layout according to `is_neox`.
- Do not call `torch.cat`, `torch.stack`, indexing assignment, PyTorch RoPE helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute RoPE rotation for `q_pe` and `k_pe` in FP32 using the FP32 `cos_sin_cache`.
- Preserve `kv_c` values exactly when concatenating into the cache.
- Cast rotated outputs back to the configured dtype only after the rotation math is complete.
- Keep `positions` and `slot_mapping` handling exact; negative slots must be ignored without modifying cache contents.
