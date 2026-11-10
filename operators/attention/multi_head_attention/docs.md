# multi_head_attention

## Function

Computes multi-head attention with learned Q, K, V, and output projections.

## Inputs

- `query`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `key`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `value`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- Do not call PyTorch attention, matmul, softmax, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute projection accumulation, QK scores, softmax max/sum, probability-value accumulation, and output projection accumulation in FP32, then cast only the final output.
