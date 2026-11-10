# batched_decode_attention

## Function

Computes single-token batched decode attention over a dense padded KV cache. Each batch item may have a different valid KV length.

## Inputs

- `q`: query tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `k_cache`: key cache tensor with shape `[batch_size, max_seq_len, num_kv_heads, head_dim]`.
- `v_cache`: value cache tensor with shape `[batch_size, max_seq_len, num_kv_heads, head_dim]`.
- `seq_lens`: INT64 tensor with shape `[batch_size]`; `seq_lens[b]` is the valid KV length for request `b`.

All floating tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[batch_size, num_q_heads, head_dim]` and dtype `dtype`.

## Implementation Notes

- This is decode attention with `q_len = 1`, not prefill attention.
- For GQA/MQA, map each query head to `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- For each batch item, only cache positions `[0, seq_lens[b])` participate in the softmax.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- Optimize for KV cache read bandwidth, head-dimension parallelism, and uneven per-request sequence lengths.
- Implement score computation, valid-length masking, softmax, and weighted value accumulation in backend-native kernel code.
- Do not call PyTorch attention, matmul, softmax, masking, or reduction operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute decode scores, softmax max/sum, and value accumulation in FP32 for each request's valid `seq_lens[b]` prefix.
