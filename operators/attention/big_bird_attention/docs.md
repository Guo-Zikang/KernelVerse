# big_bird_attention

## Function

Computes BigBird-style sparse attention with learned Q, K, V, and output projections.

Each token attends to a local sliding window, the first and last tokens act as global tokens, and each non-global token also attends to a fixed set of random positions.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, sparse attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- The PyTorch reference materializes the full `[seq_len, seq_len]` attention matrix only to make correctness semantics explicit.
- Do not blindly copy that full-matrix strategy into the target-backend implementation. Prefer computing the allowed attention regions directly: local-window links, first/last global-token links, and fixed random links.
- `window_size` controls the local regular-token window around each query token.
- `num_random_blocks` controls how many random key positions each non-global query token attends to.
- `random_seed` fixes the random position table; the generated implementation should use the same random index table semantics as the reference.
- Do not call PyTorch attention, matmul, softmax, linear, masking, random, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute sparse attention scores, softmax max/sum, and probability-value accumulation in FP32; local, global, and random masks must exclude invalid positions before the softmax reduction.
