# moe_sum_reduce

## Function

Combines top-k routed expert outputs for each token by summing over the routing dimension and applying a routing scale.

## Inputs

- `x`: tensor with shape `[num_tokens, topk, hidden_size]`.

## Output

Returns one tensor with shape `[num_tokens, hidden_size]` and dtype `dtype`.

## Implementation Notes

- Reduce the `topk` dimension for each token and hidden element.
- Multiply the sum by `routed_scaling_factor`.
- Implement the reduction and scaling in backend-native kernel code.
- Do not call PyTorch sum, reduction helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Accumulate the top-k routed expert outputs in FP32.
- Apply `routed_scaling_factor` in FP32 after the reduction.
- Cast only the final combined output back to the configured dtype.
