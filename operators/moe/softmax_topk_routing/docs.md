# softmax_topk_routing

## Function

Selects top-k experts from softmax-normalized router scores for MoE routing.

## Inputs

- `gating_output`: router logits tensor with shape `[num_tokens, num_experts]`.

## Output

Returns a tuple:

- `topk_weights`: tensor with shape `[num_tokens, topk]`.
- `topk_indices`: tensor with shape `[num_tokens, topk]`.

`topk_weights` uses the floating dtype selected by the shape entry. `topk_indices` uses `index_dtype`.

## Implementation Notes

- Compute a numerically stable softmax over the expert dimension before top-k selection.
- If `renormalize` is true, divide selected weights by their per-token selected-weight sum.
- Return selected experts in deterministic score order; when scores tie, choose the smaller expert id first.
- Do not call PyTorch softmax, topk, sort, argsort, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute the stable softmax max, exp, sum, probabilities, and renormalization sums in FP32.
- Use FP32 softmax scores for top-k selection so small FP16/BF16 rounding differences do not change selected expert ids.
- Cast only the final `topk_weights` back to the configured floating dtype; keep `topk_indices` exact in `index_dtype`.
