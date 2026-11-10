# cascade_attention

## Function

Computes decode attention for serving requests that share a common prefix and have per-request unique suffix KV states. The shared-prefix partial attention result and unique-suffix partial attention result are merged with a numerically stable log-sum-exp formula.

## Inputs

- `q`: query tensor with shape `[batch_size, num_q_heads, head_dim]`.
- `shared_k`: shared prefix key tensor with shape `[shared_len, num_kv_heads, head_dim]`.
- `shared_v`: shared prefix value tensor with shape `[shared_len, num_kv_heads, head_dim]`.
- `unique_k`: packed unique suffix key tensor with shape `[total_unique_tokens, num_kv_heads, head_dim]`.
- `unique_v`: packed unique suffix value tensor with shape `[total_unique_tokens, num_kv_heads, head_dim]`.
- `unique_indptr`: INT32 cumulative unique suffix offsets with shape `[batch_size + 1]`.

All floating tensors use the dtype selected by the shape entry.

## Output

Returns one tensor with shape `[batch_size, num_q_heads, head_dim]` and dtype `dtype`.

## Implementation Notes

- Compute a shared-prefix attention partial result for every request.
- Compute a unique-suffix attention partial result for each request using `unique_indptr`.
- Merge the two partial results with a numerically stable log-sum-exp formula.
- Requests with zero unique suffix length should use only the shared-prefix partial result.
- For GQA/MQA, map each query head to `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- Optimize for shared-prefix KV reuse, packed unique suffix offsets, uneven suffix lengths, and stable partial-result merging.
- Implement shared attention, unique attention, log-sum-exp merge, and output writeback in backend-native kernel code.
- Do not call PyTorch attention, matmul, softmax, exp, logsumexp, einsum, indexing, or reduction operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute shared and unique partial attention with FP32 online softmax state, then merge partial outputs with FP32 logsumexp weighting.
