# disentangled_attention

## Function

Computes DeBERTa-style disentangled self-attention with content and relative position interactions.

The attention score is the sum of content-to-content, content-to-position, and position-to-content scores.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement content Q/K/V projections, relative-position embedding lookup, position Q/K projections, attention score computation, softmax, value aggregation, and output projection in backend-native kernel code.
- Relative position index for query `i` and key `j` is `clamp(i - j, -max_relative_position, max_relative_position) + max_relative_position`.
- `score_c2c = q_c @ k_c`, `score_c2p` uses content query against relative position key, and `score_p2c` uses relative position query against content key.
- Combine the three score terms and apply the scale factor `head_dim ** -0.5` before softmax.
- The shape field `bias` controls whether content Q/K/V and output projection layers have bias parameters; position projections have no bias.
- Do not call PyTorch attention, matmul, einsum, embedding, softmax, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute content-position score terms, their sum, softmax max/sum, and probability-value accumulation in FP32 to avoid relative-position score drift.
