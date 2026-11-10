# cross_attention

## Function

Computes cross-attention with a query input and a separate key/value context input.

Q is projected from `query`; K and V are projected from `context`.

## Inputs

- `query`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `context`: input tensor with shape `[batch_size, context_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement query projection, context key/value projections, attention score computation over `context_len`, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- The shape field `bias` controls whether Q/K/V/O projection layers have bias parameters.
- Preserve the `forward(query, context)` signature and return one output per query position.
- Do not call PyTorch attention, matmul, softmax, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute query/context projections, QK scores over `context_len`, softmax max/sum, and probability-value accumulation in FP32, then cast only the final output.
