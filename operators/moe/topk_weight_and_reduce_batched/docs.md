# topk_weight_and_reduce_batched

## Function

Combines batched local expert outputs selected by top-k routing into one output row per token.

## Inputs

- `fused_expert_output`: local expert output tensor with shape `[num_local_experts, num_tokens, hidden_dim]`.
- `topk_weights`: routing weight tensor with shape `[num_tokens, topk]`.
- `topk_ids`: global expert id tensor with shape `[num_tokens, topk]`.

`fused_expert_output` and `topk_weights` use the dtype selected by the shape entry. `topk_ids` uses `index_dtype`.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and dtype `dtype`.

## Implementation Notes

- This is the batched local-expert layout variant, not the contiguous `[num_tokens, topk, hidden_dim]` layout.
- Expert ids in the range `[rank * num_local_experts, (rank + 1) * num_local_experts)` belong to the current rank.
- For matching local experts, read the corresponding rows from `fused_expert_output` and accumulate them into each token output.
- If `apply_router_weight_on_input` is false, multiply each matched expert output row by the corresponding `topk_weights` value before accumulating.
- If `apply_router_weight_on_input` is true, the input is already weighted; only accumulate matched rows.
- Benchmark inputs use unique top-k expert ids per token.
- Do not call PyTorch masking, indexing, count, sum, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert local expert outputs and routing weights to FP32 before multiplication.
- Accumulate matched local expert contributions in FP32.
- Cast only the final combined output back to the configured dtype.
