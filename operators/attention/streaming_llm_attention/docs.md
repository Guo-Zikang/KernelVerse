# streaming_llm_attention

## Function

Computes StreamingLLM-style sparse causal self-attention where each token attends to fixed sink tokens plus a recent causal window.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference applies Q/K/V projections, reshapes to `[batch_size, num_heads, seq_len, head_dim]`, applies a StreamingLLM mask, computes softmax attention, and applies the output projection.
- Key positions `< sink_size` are visible to every query token.
- Recent positions are visible when `key_idx <= query_idx` and `key_idx >= query_idx - window_size + 1`.
- The final visibility mask is `sink_visible OR recent_visible`.
- A high-performance target-backend implementation should avoid materializing the full `[seq_len, seq_len]` mask when practical and should compute sink-token and recent-window contributions directly.
- Do not call PyTorch attention, matmul, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute sink/recent-window scores, softmax max/sum, and probability-value accumulation in FP32; invisible positions must not contribute to the softmax reduction.
