# flash_decoding_attention

## Function

Computes flash-decoding attention over a KV cache by processing the cache in chunks and merging partial softmax results.

This operator is intended for decoding-style shapes where `query_seq_len` is small and `kv_seq_len` can be large.

## Inputs

- `query`: tensor with shape `[batch_size, num_heads, query_seq_len, head_dim]`.
- `key_cache`: tensor with shape `[batch_size, num_heads, kv_seq_len, head_dim]`.
- `value_cache`: tensor with shape `[batch_size, num_heads, kv_seq_len, head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: tensor with shape `[batch_size, num_heads, query_seq_len, head_dim]`.
- `exp_sum_stack`: per-chunk exp-sum tensor with shape `[batch_size, num_heads, query_seq_len, num_chunks]`.
- `max_stack`: per-chunk max tensor with the same shape as `exp_sum_stack`.
- `output_stack`: per-chunk normalized partial output tensor with shape `[batch_size, num_heads, query_seq_len, num_chunks, head_dim]`.

All returned tensors must use the configured output dtype.

## Implementation Notes

- Implement chunked score computation, optional causal masking, per-chunk max/exp-sum/partial-output calculation, global softmax merge, and four-output tuple generation in backend-native kernel code.
- Use `chunk_size` to split the KV cache; the final chunk may be shorter.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- For causal mode, query positions are aligned to the end of the KV cache: absolute query position is `query_index + kv_seq_len - query_seq_len`.
- Preserve the four-output tuple structure exactly.
- Do not call PyTorch attention, matmul, softmax, exp, stack, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute per-chunk max, exp sum, partial output, and global softmax merge in FP32; do not average chunk outputs directly.
