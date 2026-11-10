# multi_query_attention

## Function

Computes multi-query self-attention with learned Q, shared K, shared V, and output projections.

Unlike standard multi-head attention, all query heads attend to one shared key head and one shared value head.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q projection to `[batch_size, num_heads, seq_len, head_dim]`.
- Implement K and V projections to a single shared head with shape `[batch_size, 1, seq_len, head_dim]`.
- Each query head uses the same K/V head for scaled dot-product attention.
- Use the scale factor `head_dim ** -0.5` before softmax.
- Avoid physically materializing expanded K/V tensors when practical; broadcast or index the shared K/V head directly in the target-backend kernel.
- Do not replace this with standard multi-head attention with independent K/V heads.
- Do not call PyTorch attention, matmul, softmax, linear, expand, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute shared-K/V QK scores, softmax max/sum, and probability-value accumulation in FP32; keep query-head to shared-KV-head mapping exact.
