# static_sink_kv_populate

## Function

Populates static sink key/value entries into reserved positions of a preallocated KV cache.

## Inputs

- `sink_key`: key tensor with shape `[sink_len, num_kv_heads, head_dim]`.
- `sink_value`: value tensor with shape `[sink_len, num_kv_heads, head_dim]`.
- `kv_cache_k`: key cache tensor with shape `[total_slots, num_kv_heads, head_dim]`.
- `kv_cache_v`: value cache tensor with shape `[total_slots, num_kv_heads, head_dim]`.
- `config`: int32 tensor with values `[block_size, sink_len]`.

All key/value and cache tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `kv_cache_k`: updated key cache with shape `[total_slots, num_kv_heads, head_dim]`.
- `kv_cache_v`: updated value cache with shape `[total_slots, num_kv_heads, head_dim]`.

Both outputs use the configured shape dtype.

## Implementation Notes

- Sink tokens are written to contiguous slots `[block_size, block_size + sink_len)`.
- Copy `sink_key[i]` and `sink_value[i]` to the corresponding sink slot.
- Implement the cache update in backend-native kernel code.
- Do not call PyTorch advanced indexing, arange, scatter, index assignment, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache write/copy operator; preserve source sink key/value values exactly in the configured dtype.
- Do not rescale, normalize, or convert cache values through FP32 for pure writes.
- Keep slot calculation exact and preserve cache entries outside the sink range.
