# apply_shuffle_mul_sum

## Function

Merges routed MoE expert outputs by shuffling rows, optionally applying routing factors, and summing over each token's top-k expert outputs.

## Inputs

- `x`: expert output tensor with shape `[m * topk, hidden_size]`.
- `permutation`: row permutation tensor with shape `[m * topk]`.
- `factors`: routing factor tensor with shape `[m * topk]`.

`x` and `factors` use the dtype selected by the shape entry. `permutation` uses the selected `index_dtype`.

## Output

Returns one tensor with shape `[m, hidden_size]` and the configured shape dtype.

## Implementation Notes

- Implement row gather, optional factor multiply, and top-k reduction in backend-native kernel code.
- If `has_factors` is false, ignore factor values and only reduce gathered rows.
- Avoid materializing the full `[m, topk, hidden_size]` intermediate when possible.
- Do not call `index_select`, `view`, `sum`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert gathered expert outputs and routing factors to FP32 before multiplication.
- Accumulate routed top-k expert contributions in FP32.
- Cast only the final combined output back to the configured dtype.
