# append_attention

## Function

Appends one new K/V token per request into a dense padded KV cache, then computes single-token decode attention over the updated cache.

## Inputs

- `q`: query tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `new_k`: new key tensor with shape `[batch_size, num_kv_heads, head_dim]`.
- `new_v`: new value tensor with shape `[batch_size, num_kv_heads, head_dim]`.
- `k_cache`: key cache tensor with shape `[batch_size, max_seq_len, num_kv_heads, head_dim]`.
- `v_cache`: value cache tensor with shape `[batch_size, max_seq_len, num_kv_heads, head_dim]`.
- `seq_lens`: INT64 tensor with shape `[batch_size]`; `seq_lens[b]` is the KV length before append.

All floating tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: decode attention output with shape `[batch_size, num_q_heads, head_dim]`.
- `updated_k_cache`: key cache after appending `new_k`.
- `updated_v_cache`: value cache after appending `new_v`.

All returned tensors use dtype `dtype`.

## Implementation Notes

- Write `new_k[b]` and `new_v[b]` to cache position `seq_lens[b]`.
- Compute attention over valid positions `[0, seq_lens[b] + 1)` after the append.
- For GQA/MQA, map each query head to `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- Optimize for the fused append-and-attend decode step instead of treating cache write and attention as unrelated operations.
- Implement cache append, valid-length masking, score computation, softmax, and value accumulation in backend-native kernel code.
- Do not call PyTorch indexing, scatter, attention, matmul, softmax, masking, or reduction operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, keep cache writes in the requested dtype but compute decode scores, softmax max/sum, and value accumulation in FP32 over the valid cache prefix.
