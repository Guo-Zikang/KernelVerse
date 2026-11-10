# concat_and_cache_mla

## Function

Concatenates MLA latent KV features and RoPE key features, then writes the fused entries into paged KV cache slots.

This is an MLA KV cache update operator, not a full attention operator.

## Inputs

- `kv_c`: latent KV tensor with shape `[num_tokens, latent_dim]`.
- `k_pe`: key positional feature tensor with shape `[num_tokens, rope_dim]`.
- `kv_cache`: cache tensor with shape `[num_blocks, block_size, latent_dim + rope_dim]`.
- `slot_mapping`: physical cache slot indices with shape `[num_tokens]`; negative slots are ignored.

`kv_c`, `k_pe`, and `kv_cache` use the dtype selected by the shape entry. `slot_mapping` is INT64.

## Output

Returns the updated KV cache tensor with shape `[num_blocks, block_size, latent_dim + rope_dim]`.

The output uses the dtype selected by the shape entry.

## Implementation Notes

- Implement latent/key concatenation, slot filtering, and cache write in backend-native kernel code.
- Treat `slot_mapping` as indexing the flattened physical cache slots over `num_blocks * block_size`.
- Do not call `torch.cat`, indexing assignment, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Concatenate `kv_c` and `k_pe` without changing their numerical values.
- Preserve both latent KV and positional feature values exactly in the configured dtype.
- Keep `slot_mapping` handling exact; negative slots must be ignored without modifying cache contents.
