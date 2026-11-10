# etc_global_local_attention

## Function

Computes ETC-style global-local sparse attention with learned Q, K, V, and output projections.

Regular tokens attend to a local window of regular tokens and to all global tokens. Global tokens attend to all regular and global tokens.

## Inputs

- `regular_tokens`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `global_tokens`: tensor with shape `[batch_size, num_global_tokens, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- `updated_regular`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `updated_global`: tensor with shape `[batch_size, num_global_tokens, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K/V projections, sparse attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Use the scale factor `head_dim ** -0.5` before softmax.
- The PyTorch reference materializes the full `[seq_len + num_global_tokens, seq_len + num_global_tokens]` attention matrix only to make correctness semantics explicit.
- Do not blindly copy that full-matrix strategy into the target-backend implementation. Prefer computing the allowed attention regions directly: regular-to-local-regular, regular-to-global, global-to-regular, and global-to-global.
- Regular-token attention includes the local regular-token window and all global tokens.
- Global-token attention includes all regular tokens and all global tokens.
- Do not call PyTorch attention, matmul, softmax, linear, masking, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute regular-local, regular-global, global-regular, and global-global scores plus softmax max/sum and value accumulation in FP32.
