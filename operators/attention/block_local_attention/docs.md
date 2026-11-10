# block_local_attention

## Function

Computes block-local attention with learned Q, K, V, and output projections.

The sequence is split into fixed-size blocks, and each token attends only to tokens in the same block.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, per-block attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- `seq_len` is divisible by `block_size`; each contiguous block is independent.
- Do not compute attention across different blocks.
- Do not call PyTorch attention, matmul, softmax, linear, reshape-heavy helper logic, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute per-block QK scores, softmax max/sum, and probability-value accumulation in FP32, then cast only the final output to the configured dtype.
