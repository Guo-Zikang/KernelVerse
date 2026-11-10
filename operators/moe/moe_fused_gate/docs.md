# moe_fused_gate

## Function

Computes DeepSeek-style grouped MoE routing with optional fused shared experts.

## Inputs

- `scores`: router score tensor with shape `[num_tokens, num_experts]`.
- `bias`: expert bias tensor with shape `[num_experts]`.

Both tensors use the dtype selected by the shape entry.

## Output

Returns a tuple:

- `topk_weights`: selected expert weights with shape `[num_tokens, topk]`.
- `topk_ids`: selected expert ids with shape `[num_tokens, topk]`.

`topk_weights` uses `dtype`. `topk_ids` uses `index_dtype`.

## Implementation Notes

- Apply sigmoid to `scores` before routing.
- Add `bias` only for group and expert selection.
- Split experts into `num_expert_group` groups; each group contains `num_experts / num_expert_group` experts.
- For each token, score each expert group by summing the top 2 biased scores inside that group.
- Select `topk_group` groups, mask out all experts outside those groups, then select routed experts.
- Break group ties by smaller group id first, and expert ties by smaller expert id first.
- If `num_fused_shared_experts > 0`, append shared expert ids after routed experts. Shared expert ids start at `num_experts`.
- Shared expert weights are derived from the routed weight sum divided by `2.5`.
- Renormalize routed expert weights by the routed weight sum.
- Implement sigmoid, grouped top-k, expert top-k, shared expert append, renormalization, and writeback in backend-native kernel code.
- Do not call PyTorch sigmoid, topk, scatter, gather, masked_fill, view/reshape helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute sigmoid scores, bias addition, group scores, routed expert scores, and shared expert weights in FP32.
- Use FP32 scores for both group top-k and expert top-k selection so small FP16/BF16 rounding differences do not change selected ids.
- Apply routed weight sums and renormalization in FP32, then cast only final `topk_weights` back to the configured dtype.
