# causal_multi_head_attention

## Function

Computes causal multi-head self-attention with learned Q, K, V, and output projections.

Each token attends only to itself and previous sequence positions.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, causal attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- Apply a causal lower-triangular mask: query position `i` must not attend to key positions `j > i`.
- The shape field `bias` controls whether the projection layers have bias parameters.
- Avoid materializing unnecessary full causal masks when practical.
- Do not call PyTorch attention, matmul, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute projection accumulation, causal QK scores, softmax max/sum, and probability-value accumulation in FP32; future positions must not contribute to the softmax reduction.
