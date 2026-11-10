# ragged_kvcache_attention

## Function

Computes single-token decode attention over a packed ragged KV cache. Each request's KV cache is stored as one contiguous segment in a shared packed buffer.

## Inputs

- `q`: query tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `k_cache`: packed key cache tensor with shape `[total_kv_tokens, num_kv_heads, head_dim]`.
- `v_cache`: packed value cache tensor with shape `[total_kv_tokens, num_kv_heads, head_dim]`.
- `kv_indptr`: INT32 cumulative KV offsets with shape `[batch_size + 1]`.

All floating tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[batch_size, num_q_heads, head_dim]` and dtype `dtype`.

## Implementation Notes

- This is ragged KV-cache decode attention, not paged KV-cache attention and not dense padded KV-cache attention.
- For request `b`, attend only over cache positions `[kv_indptr[b], kv_indptr[b + 1])`.
- For GQA/MQA, map each query head to `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- Optimize for prefix-sum offset handling, contiguous ragged KV reads, uneven sequence lengths, and per-request softmax reduction.
- Implement indptr lookup, score computation, softmax, and weighted value accumulation in backend-native kernel code.
- Do not call PyTorch attention, matmul, softmax, exp, einsum, indexing, masking, or reduction operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute ragged-cache scores, softmax max/sum, and value accumulation in FP32 for each `[kv_indptr[b], kv_indptr[b + 1])` segment.
