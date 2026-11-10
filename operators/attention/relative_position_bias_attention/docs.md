# relative_position_bias_attention

## Function

Computes multi-head self-attention with T5-style learned relative position bucket bias added to attention scores before softmax.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference applies Q/K/V projections, reshapes to `[batch_size, num_heads, seq_len, head_dim]`, and computes scaled dot-product attention.
- Relative positions are `context_position - memory_position`.
- Buckets are split by sign: negative relative positions use the upper half of the bucket range.
- Small distances use exact buckets; larger distances use logarithmic buckets capped at `num_buckets // 2 - 1` within each sign half.
- `relative_attention_bias` stores one learned bias value per bucket and head. Gather the bucket bias and add it to scores before softmax.
- Do not call PyTorch attention, matmul, softmax, embedding, bucket generation, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute QK scores, relative-bias addition, softmax max/sum, and probability-value accumulation in FP32; bucket indices must stay exact.
