# batched_prefill_attention

## Function

Computes packed variable-length causal prefill attention for LLM serving. Each batch item is described by query and KV indptr arrays, so prompt lengths can differ across requests.

## Inputs

- `q`: query tensor with shape `[total_q_tokens, num_q_heads, head_dim]`.
- `k`: key tensor with shape `[total_kv_tokens, num_kv_heads, head_dim]`.
- `v`: value tensor with shape `[total_kv_tokens, num_kv_heads, head_dim]`.
- `qo_indptr`: INT32 cumulative query offsets with shape `[batch_size + 1]`.
- `kv_indptr`: INT32 cumulative KV offsets with shape `[batch_size + 1]`.

All floating tensors use the dtype selected by the shape entry.

## Output

Returns one packed output tensor with shape `[total_q_tokens, num_q_heads, head_dim]` and dtype `dtype`.

## Implementation Notes

- This is serving prefill attention, not single-token decode attention.
- For each request, compute attention only within the matching query and KV slices from `qo_indptr` and `kv_indptr`.
- For GQA/MQA, map each query head to `kv_head = q_head // (num_q_heads // num_kv_heads)`.
- Apply causal masking inside each request. When `q_len < kv_len`, the query slice represents the last `q_len` positions of that KV context.
- Use `softmax_scale` when present, otherwise use `head_dim ** -0.5`.
- Optimize for indptr handling, variable prompt lengths, block-level softmax reduction, and uneven batch workloads.
- Implement score computation, causal masking, softmax, and weighted value accumulation in backend-native kernel code.
- Do not call PyTorch attention, matmul, softmax, exp, einsum, masking, or reduction operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, use FP32 online softmax state for each packed request: score accumulation, running max, exp sum, and value accumulator should stay FP32 until final output cast.
