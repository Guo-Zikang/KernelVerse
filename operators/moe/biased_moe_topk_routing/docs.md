# biased_moe_topk_routing

## Function

Computes MoE router scores, applies a per-expert correction bias for expert selection, and returns selected expert weights and ids.

## Inputs

- `gating_output`: router logits tensor with shape `[num_tokens, num_experts]`.
- `e_score_correction_bias`: tensor with shape `[num_experts]`.

The routing decision is computed from `gating_output` and `e_score_correction_bias`.

## Output

Returns a tuple:

- `topk_weights`: tensor with shape `[num_tokens, topk]`.
- `topk_ids`: tensor with shape `[num_tokens, topk]`.

All floating input and weight output tensors use the dtype selected by the shape entry. `topk_ids` uses `index_dtype`.

## Implementation Notes

- For `scoring_func="softmax"`, compute a numerically stable softmax over the expert dimension before top-k selection.
- For `scoring_func="sigmoid"`, apply sigmoid to each expert logit before top-k selection.
- Add `e_score_correction_bias` to the computed scores only for choosing `topk_ids`.
- Gather `topk_weights` from the original scores, not from the bias-corrected scores.
- Return selected experts in deterministic bias-corrected score order; when scores tie, choose the smaller expert id first.
- If `renormalize` is true, divide selected weights by their per-token selected-weight sum.
- Do not call PyTorch softmax, sigmoid, topk, gather, sort, argsort, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute router scores, sigmoid/softmax, correction bias addition, and renormalization sums in FP32.
- Use FP32 bias-corrected scores for top-k selection so small FP16/BF16 rounding differences do not change selected expert ids.
- Cast only the final `topk_weights` back to the configured floating dtype; keep `topk_ids` exact in `index_dtype`.
