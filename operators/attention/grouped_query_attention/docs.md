# grouped_query_attention

## Function

Computes grouped-query self-attention with learned Q, grouped K, grouped V, and output projections.

Grouped-query attention uses fewer key/value heads than query heads. Each K/V head is shared by `num_heads / num_kv_heads` query heads.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q projection to `[batch_size, num_heads, seq_len, head_dim]`.
- Implement K and V projections to `[batch_size, num_kv_heads, seq_len, head_dim]`.
- Query head `h` should use K/V head `h // (num_heads / num_kv_heads)`.
- Use the scale factor `head_dim ** -0.5` before softmax.
- Avoid physically materializing repeated K/V heads when practical; map query heads to their shared K/V group directly in the target-backend kernel.
- Do not replace this with standard multi-head attention with independent K/V heads.
- Do not call PyTorch attention, matmul, softmax, linear, repeat_interleave, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute grouped QK scores, softmax max/sum, and probability-value accumulation in FP32; keep query-head to shared-KV-head mapping exact.
