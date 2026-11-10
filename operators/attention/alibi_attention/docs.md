# alibi_attention

## Function

Computes multi-head self-attention with ALiBi linear distance bias added to attention scores before softmax.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference applies Q/K/V projections, reshapes to `[batch_size, num_heads, seq_len, head_dim]`, and computes scaled dot-product attention.
- ALiBi slopes are generated as `slope[h] = 2 ** (-8 * (h + 1) / num_heads)`.
- ALiBi bias is `bias[0, h, i, j] = -slope[h] * abs(i - j)`.
- Add the ALiBi bias to attention scores before softmax.
- This is the full attention operator. The separate `alibi` operator only generates the positional bias tensor.
- Do not call PyTorch attention, matmul, softmax, linear, bias generation, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute Q/K/V projection accumulation, ALiBi-biased scores, softmax max/sum, and probability-value accumulation in FP32, then cast only the final output to the configured dtype.
