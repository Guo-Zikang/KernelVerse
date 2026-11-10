# rmsnorm_gated

## Function

Adds a residual tensor to the input tensor, applies RMSNorm over the hidden dimension, then modulates the result with a Swish gate.

## Inputs

- `x`: input tensor with shape `[batch_size, hidden_dim]`.
- `gate_input`: gate tensor with shape `[batch_size, hidden_dim]`.
- `weight`: RMSNorm weight with shape `[hidden_dim]`.
- `residual`: residual tensor with shape `[batch_size, hidden_dim]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `eps` and `prenorm` are known from the shape entry.

## Output

- If `prenorm` is false, returns one tensor with shape `[batch_size, hidden_dim]`.
- If `prenorm` is true, returns `(output, residual_out)`, both with shape `[batch_size, hidden_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement residual addition, RMSNorm, Swish gating, and elementwise multiplication in backend-native kernel code.
- Do not call `torch.rsqrt`, `torch.mean`, `torch.pow`, `torch.sigmoid`, RMSNorm libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute `x + residual`, RMS variance, `rsqrt`, normalization, and weight scaling in FP32.
- Compute the gate path `gate * sigmoid(gate)` in FP32 before multiplying with the normalized value.
- Cast final outputs back to the configured dtype; for `prenorm=True`, return the residual output with the same dtype semantics as the reference.
