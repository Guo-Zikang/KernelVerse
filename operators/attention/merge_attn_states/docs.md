# merge_attn_states

## Function

Merges prefix and suffix attention outputs using numerically stable log-sum-exp weights.

## Inputs

- `prefix_output`: tensor with shape `[num_tokens, num_heads, head_size]`.
- `prefix_lse`: tensor with shape `[num_tokens, num_heads]`.
- `suffix_output`: tensor with shape `[num_tokens, num_heads, head_size]`.
- `suffix_lse`: tensor with shape `[num_tokens, num_heads]`.

Output tensors use the dtype selected by the shape entry. LSE tensors use FP32.

## Output

- Tensor with shape `[num_tokens, num_heads, head_size]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Use `max(prefix_lse, suffix_lse)` before exponentiation to keep the merge numerically stable.
- Compute weights from the prefix and suffix LSE values, then blend the corresponding output vectors.
- Implement max, exp, weight normalization, vector blend, and writeback in backend-native kernel code.
- Do not call PyTorch maximum, exp, broadcasting helpers, attention libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute max, exp weights, denominator, and vector blending in FP32; cast only the merged value output to the configured dtype.
