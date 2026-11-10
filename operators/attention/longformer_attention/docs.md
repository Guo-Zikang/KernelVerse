# longformer_attention

## Function

Computes Longformer-style sparse self-attention with local sliding windows and global tokens.

Each token attends to a centered local window. Tokens listed in `global_indices` attend to all tokens, and all tokens attend to every global token.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, local-window attention, global-token attention, softmax, value aggregation, and output projection in backend-native kernel code.
- Local window radius is `window_size // 2`, so token `i` attends to positions `[i - radius, i + radius]` clipped to sequence bounds.
- For each global index `g`, row `g` and column `g` are fully visible.
- The PyTorch reference materializes the full `[seq_len, seq_len]` mask for clarity. A target-backend implementation should compute local-window and global-token positions directly when practical.
- Do not call PyTorch attention, matmul, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute local/global attention scores, softmax max/sum, and probability-value accumulation in FP32; invalid positions must not contribute to the reduction.
