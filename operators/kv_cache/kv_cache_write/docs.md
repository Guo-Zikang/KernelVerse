# kv_cache_write

## Function

Writes current key and value tensors into preallocated key/value cache tensors at selected sequence positions.

## Inputs

- `key_cache`: key cache tensor with shape `[batch_size, num_kv_heads, max_seq_len, head_dim]`.
- `value_cache`: value cache tensor with shape `[batch_size, num_kv_heads, max_seq_len, head_dim]`.
- `key`: key tensor with shape `[batch_size, num_kv_heads, write_seq_len, head_dim]`.
- `value`: value tensor with shape `[batch_size, num_kv_heads, write_seq_len, head_dim]`.
- `positions`: write positions with shape `[write_seq_len]` for `shared` mode or `[batch_size, write_seq_len]` for `per_batch` mode.

All cache and value tensors use the dtype selected by the shape entry. `positions` uses integer dtype.

## Output

Returns a tuple:

- `key_cache`: updated key cache with the same shape as the input key cache.
- `value_cache`: updated value cache with the same shape as the input value cache.

Both outputs use the configured shape dtype.

## Implementation Notes

- For `position_mode="shared"`, every batch item writes to the same `positions`.
- For `position_mode="per_batch"`, each batch item writes to its own row of `positions`.
- Implement the cache writes for key and value tensors in backend-native kernel code.
- Do not call PyTorch advanced indexing, scatter, index assignment, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache write/copy operator; preserve source key/value values exactly in the configured dtype.
- Do not rescale, normalize, or convert cache values through FP32 for pure writes.
- Keep position/index handling exact and preserve cache entries that are not written.
