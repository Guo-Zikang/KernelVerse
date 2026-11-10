# flash_attention

## Function

Computes non-causal self-attention with learned Q, K, V, and output projections using a tiled online-softmax reference.

The reference also returns block-level trace tensors from the tiled computation.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `output`: tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- `block_maxes`: block trace tensor with shape `[batch_size, num_heads, num_q_blocks, num_kv_blocks, block_q]`.
- `block_exp_sums`: block trace tensor with the same shape as `block_maxes`.
- `block_output_sums`: block trace tensor with the same shape as `block_maxes`.

All returned tensors must use the configured output dtype.

## Implementation Notes

- Implement Q/K/V projections, tiled score computation, online softmax accumulation, value aggregation, output projection, and block trace outputs in backend-native kernel code.
- Use `block_size_q` and `block_size_kv` to define query and key/value tiles; the last tile may be shorter.
- Use the scale factor `head_dim ** -0.5` before softmax.
- Preserve the four-output tuple structure exactly.
- `block_maxes`, `block_exp_sums`, and `block_output_sums` are diagnostic trace outputs from each query/key-value tile pair.
- This reference is non-causal; do not apply a causal mask.
- Do not call PyTorch attention, matmul, softmax, exp, linear, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, keep tiled QK scores, online softmax max/sum state, probability-value accumulation, and output projection accumulation in FP32.
- Trace tensors must follow the tiled reference semantics; compute `block_maxes`, `block_exp_sums`, and `block_output_sums` from the same FP32 intermediate path.
