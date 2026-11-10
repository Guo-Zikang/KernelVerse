# self_attention

## Function

Computes non-causal multi-head self-attention with learned Q, K, V, and output projections.

The same input tensor is used as the source for Q, K, and V.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, full non-causal attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- The shape field `bias` controls whether Q/K/V/O projection layers have bias parameters.
- This is full bidirectional self-attention; do not apply a causal mask.
- Do not call PyTorch attention, matmul, softmax, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute projection accumulation, full QK scores, softmax max/sum, probability-value accumulation, and output projection accumulation in FP32, then cast only the final output.
