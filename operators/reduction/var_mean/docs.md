# var_mean

## Function

Computes variance and mean along one or more dimensions, equivalent to `torch.var_mean`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- `variance`: tensor reduced along `dim`.
- `mean`: tensor reduced along `dim`.
- If `keepdim` is true, reduced dimensions remain with size 1.
- Output dtypes must match the configured shape dtype.

## Implementation Notes

- Implement both variance and mean reductions in backend-native kernel code.
- Support positive and negative `dim` values.
- Support multi-axis reduction when `dim` is a list.
- Support `correction` values from the shape config. The denominator is `N - correction`.
- For FP16/BF16 inputs, compute the mean, squared differences, variance accumulation, and denominator scaling in FP32.
- Prefer a numerically stable two-pass algorithm or Welford-style accumulation; avoid computing variance as `E[x^2] - E[x]^2` in low precision.
- Do not call `torch.var_mean`, `torch.var`, `torch.mean`, or other PyTorch operators for the core computation in `ModelNew.forward`.
