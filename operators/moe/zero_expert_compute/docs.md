# zero_expert_compute

## Function

Computes the identity zero-expert contribution for routed MoE tokens.

## Inputs

- `expert_indices`: integer tensor with shape `[num_tokens, topk]`.
- `expert_scales`: tensor with shape `[num_tokens, topk]` and dtype `dtype`.
- `hidden_states`: tensor with shape `[num_tokens, hidden_dim]` and dtype `dtype`.

## Output

Returns one tensor with shape `[num_tokens, hidden_dim]` and dtype `dtype`.

## Implementation Notes

- Expert ids smaller than `num_experts` are regular experts and contribute zero in this operator.
- Expert ids greater than or equal to `num_experts` are identity zero-expert routes.
- Sum the routing scales for identity zero-expert routes per token, then multiply the token's `hidden_states` by that summed scale.
- Implement the mask, scale reduction, and elementwise multiply in backend-native kernel code.
- Do not call PyTorch masking, where, sum, or multiply operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Accumulate identity-route scales in FP32 before multiplying `hidden_states`.
- Perform the hidden-state scaling in FP32 when practical.
- Cast only the final output back to the configured dtype.
