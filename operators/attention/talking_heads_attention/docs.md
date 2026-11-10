# talking_heads_attention

## Function

Computes talking-heads self-attention, which mixes attention heads before and after the softmax.

## Inputs

- `tensor`: input tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `[batch_size, seq_len, num_heads * head_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- The reference computes Q/K/V projections, reshapes to `[batch_size, num_heads, seq_len, head_dim]`, and computes scaled dot-product scores.
- Before softmax, scores are mixed across the head dimension using `pre_softmax_head_mix`.
- After softmax, attention probabilities are mixed across the head dimension using `post_softmax_head_mix`.
- The head mixing rule is equivalent to `torch.einsum("bhij,gh->bgij", tensor, weight)`: output head `g` is a weighted sum over input heads `h` for every `[query, key]` position.
- The mixed probabilities are multiplied by values, then the output projection `w_o` is applied.
- Do not call PyTorch attention, matmul, einsum, softmax, linear, reshape/permutation, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute score head-mixing, softmax max/sum, probability head-mixing, and probability-value accumulation in FP32, then cast only the final output.
