# causal_sliding_window_attention

## Function

Computes causal sliding-window self-attention with learned Q, K, V, and output projections.

For query position `i`, only key positions `j` are visible when `j <= i` and `j >= i - window_size`.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, causal sliding-window attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- `window_size` is the maximum backward distance from a query position, so each query can attend to up to `window_size + 1` keys including itself.
- Do not compute attention to future positions or to past positions outside the sliding window.
- Avoid materializing a full `[seq_len, seq_len]` mask when practical; compute the valid key range directly in the target-backend kernel.
- Do not call PyTorch attention, matmul, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute windowed QK scores, softmax max/sum, and probability-value accumulation in FP32; out-of-window and future positions must not contribute to the softmax reduction.
