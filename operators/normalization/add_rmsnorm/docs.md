# add_rmsnorm

## Function

Adds a residual tensor to the input tensor, then applies RMSNorm over the last dimension.

## Inputs

- `x`: input tensor with shape `shape`.
- `residual`: residual tensor with the same shape as `x`.
- `weight`: RMSNorm weight with shape `[shape[-1]]`.

All input tensors use the dtype selected by the shape entry.

## Parameters

- `eps` is known from the shape entry.

## Output

- Tensor with the same shape as `x`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement residual addition, RMS statistic computation, normalization, and weight scaling in backend-native kernel code.
- Do not call `torch.rsqrt`, `torch.mean`, `torch.pow`, RMSNorm libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute `x + residual` in FP32 before RMS statistic computation.
- Compute `mean((x + residual) * (x + residual))`, `rsqrt`, normalization, and weight scaling in FP32.
- Cast only the final output back to the configured dtype.
