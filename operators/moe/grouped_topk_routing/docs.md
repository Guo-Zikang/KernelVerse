# grouped_topk_routing

## Function

Computes MoE router scores, selects top expert groups, then selects the final top-k experts from the selected groups.

## Inputs

- `gating_output`: router logits tensor with shape `[num_tokens, num_experts]`.
- `e_score_correction_bias`: optional correction bias tensor with shape `[num_experts]`.

The correction bias is used only when `has_correction_bias` is true.

## Output

Returns a tuple:

- `topk_weights`: tensor with shape `[num_tokens, topk]`.
- `topk_ids`: tensor with shape `[num_tokens, topk]`.

`topk_weights` uses `weight_dtype`. `topk_ids` uses `index_dtype`.

## Implementation Notes

- For `scoring_func="softmax"`, compute a numerically stable softmax over the expert dimension before group selection.
- For `scoring_func="sigmoid"`, apply sigmoid to each expert logit before group selection.
- Select `topk_group` expert groups before selecting final `topk` experts.
- If `has_correction_bias` is true, use correction bias for group and expert selection, but gather final weights from the original scores.
- Break group ties by smaller group id first, and expert ties by smaller expert id first.
- If `renormalize` is true, divide selected weights by their per-token selected-weight sum.
- Apply `routed_scaling_factor` after optional renormalization.
- Do not call PyTorch softmax, sigmoid, topk, sort, argsort, scatter, gather, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute router scores, sigmoid/softmax, correction bias addition, group scores, and expert scores in FP32.
- Use FP32 scores for both group top-k and expert top-k selection so small FP16/BF16 rounding differences do not change selected ids.
- Apply renormalization and `routed_scaling_factor` in FP32, then cast only final `topk_weights` back to `weight_dtype`.
