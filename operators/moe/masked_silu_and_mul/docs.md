# masked_silu_and_mul

## Function

Applies SiLU gated activation to ragged MoE expert batches.

## Inputs

- `x`: tensor with shape `[num_experts, max_tokens, 2 * hidden_dim]`.
- `token_counts`: INT32 tensor with shape `[num_experts]`.

For each expert, only tokens in `[0, token_counts[expert])` are valid.

## Output

Returns one tensor with shape `[num_experts, max_tokens, hidden_dim]` and dtype `dtype`.

Padding positions outside each expert's valid token range are zero.

## Implementation Notes

- Split the last dimension of each valid token into `gate` and `up`.
- Compute `silu(gate) * up`.
- Skip padded tokens where token index is greater than or equal to `token_counts[expert]`.
- Write zero for padded output positions.
- Implement masking, SiLU activation, multiply, and writeback in backend-native kernel code.
- Do not call PyTorch chunk, silu, sigmoid, gather/scatter helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute sigmoid/SiLU for the gate branch in FP32.
- Perform the gate-up multiplication in FP32 for valid tokens when practical.
- Cast only valid-token outputs back to the configured dtype; padded output positions remain exact zero.
