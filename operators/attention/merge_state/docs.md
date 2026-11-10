# merge_state

## Function

Merges two attention states and their log-sum-exp scores using numerically stable weights.

For each token and head:

```text
max_lse = max(s_a, s_b)
p = exp(s_a - max_lse)
q = exp(s_b - max_lse)
denom = p + q
v_merged = v_a * p / denom + v_b * q / denom
s_merged = log(denom) + max_lse
```

## Inputs

- `v_a`: value state tensor with shape `[num_tokens, num_heads, head_size]`.
- `s_a`: LSE state tensor with shape `[num_tokens, num_heads]`.
- `v_b`: value state tensor with shape `[num_tokens, num_heads, head_size]`.
- `s_b`: LSE state tensor with shape `[num_tokens, num_heads]`.

Value tensors use the dtype selected by the shape entry. LSE tensors use FP32.

## Output

Returns a tuple:

- `v_merged`: tensor with shape `[num_tokens, num_heads, head_size]`.
- `s_merged`: tensor with shape `[num_tokens, num_heads]`.

`v_merged` uses the configured shape dtype. `s_merged` uses FP32.

## Implementation Notes

- Use `max(s_a, s_b)` before exponentiation to keep the merge numerically stable.
- `head_size` is a multiple of 4, enabling efficient target-backend vector packing.
- Implement max, exp, log, weight normalization, vector blend, and writeback in backend-native kernel code.
- Do not call PyTorch maximum, exp, log, broadcasting helpers, attention libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute max, exp weights, denominator, merged LSE, and vector blending in FP32; cast only `v_merged` to the configured dtype while keeping `s_merged` FP32.
