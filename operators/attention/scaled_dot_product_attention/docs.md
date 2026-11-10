# scaled_dot_product_attention

## Function

Computes scaled dot-product attention:

`softmax(query @ key^T / sqrt(head_dim)) @ value`

## Inputs

- `query`: tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `key`: tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- `value`: tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, num_heads, seq_len, head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Compute attention scores as `query @ key.transpose(-2, -1)`.
- Scale scores by `head_dim ** -0.5`.
- Apply softmax over the key sequence dimension.
- Multiply attention weights by `value`.
- Do not call PyTorch attention, scaled_dot_product_attention, matmul, softmax, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute QK scores, softmax max/sum, and probability-value accumulation in FP32, then cast only the final output to the configured dtype.
