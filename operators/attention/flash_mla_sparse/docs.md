# flash_mla_sparse

## Function

Computes sparse MLA-style attention over selected key/value positions. For each query token and head, `indices` provides the candidate KV positions used for the local softmax and weighted sum.

## Inputs

- `q`: query tensor with shape `[seq_q, num_heads, head_dim]`.
- `kv`: key/value tensor with shape `[seq_kv, num_heads, head_dim]`.
- `indices`: int32 tensor with shape `[seq_q, num_heads, topk]`.

All Q/KV tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: sparse attention output with shape `[seq_q, num_heads, head_dim]`.
- `max_logits`: maximum score per query/head with shape `[seq_q, num_heads]`.
- `lse`: log-sum-exp per query/head with shape `[seq_q, num_heads]`.

The output tensor uses the configured shape dtype. `max_logits` and `lse` are returned as FP32.

## Implementation Notes

- Implement the gather by `indices`, dot products, softmax, weighted sum, max logits, and log-sum-exp in custom target-backend kernels.
- Use the scale factor from `softmax_scale` when provided; otherwise use `head_dim ** -0.5`.
- Do not call PyTorch attention, gather, matmul, softmax, exp, einsum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute gathered QK dot products, sparse softmax max/sum, value accumulation, `max_logits`, and `lse` in FP32.
