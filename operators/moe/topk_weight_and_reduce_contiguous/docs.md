# topk_weight_and_reduce_contiguous

## Function

Combines contiguous top-k routed expert outputs into one output row per token.

## Inputs

- `fused_expert_output`: expert output tensor with shape `[num_tokens, topk, hidden_dim]` or flattened shape `[num_tokens * topk, hidden_dim]`.
- `topk_weights`: routing weight tensor with shape `[num_tokens, topk]`.
- `topk_ids`: expert id tensor with shape `[num_tokens, topk]`.

`fused_expert_output` and `topk_weights` use the dtype selected by the shape entry. `topk_ids` uses `index_dtype`.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and dtype `dtype`.

## Implementation Notes

- Treat `fused_expert_output` as a contiguous top-k layout for each token.
- If `apply_router_weight_on_input` is false, multiply each expert output row by its corresponding `topk_weights` value before reducing.
- If `apply_router_weight_on_input` is true, the input is already weighted; only reduce across the top-k dimension.
- `topk_ids` is part of the interface but is not needed for addressing in the contiguous layout.
- Do not call PyTorch view, multiplication helpers, sum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert expert outputs and routing weights to FP32 before multiplication.
- Accumulate routed top-k expert contributions in FP32.
- Cast only the final combined output back to the configured dtype.
