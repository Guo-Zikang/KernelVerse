# moe_topk_routing

## Function

Computes MoE router scores, selects the top-k experts per token, and returns routing metadata for downstream expert dispatch.

## Inputs

- `gating_output`: router logits tensor with shape `[num_tokens, num_experts]`.

The top-k routing decision is computed from `gating_output`.

## Output

Returns a tuple:

- `topk_weights`: tensor with shape `[num_tokens, topk]`.
- `topk_ids`: tensor with shape `[num_tokens, topk]`.
- `token_expert_indices`: tensor with shape `[num_tokens, topk]`.

`topk_weights` uses `weight_dtype`. `topk_ids` and `token_expert_indices` use `index_dtype`.

## Implementation Notes

- For `scoring_func="softmax"`, compute a numerically stable softmax over the expert dimension before top-k selection.
- For `scoring_func="sigmoid"`, apply sigmoid to each expert logit before top-k selection.
- Select top-k experts along the last dimension of `gating_output`.
- Return selected experts in deterministic score order; when scores tie, choose the smaller expert id first.
- If `renormalize` is true, divide selected weights by their per-token selected-weight sum.
- `token_expert_indices` is a row-major range reshaped to `[num_tokens, topk]`.
- Do not call PyTorch softmax, sigmoid, topk, sort, argsort, arange, reshape, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute router scores, sigmoid/softmax, and renormalization sums in FP32.
- Use FP32 scores for top-k selection so small FP16/BF16 rounding differences do not change selected expert ids.
- Cast only the final `topk_weights` back to `weight_dtype`; keep `topk_ids` and `token_expert_indices` exact in `index_dtype`.
