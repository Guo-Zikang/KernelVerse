# logsumexp

## Function

Computes the logarithm of summed exponentials along one or more dimensions, equivalent to `torch.logsumexp`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor reduced along `dim`.
- If `keepdim` is true, reduced dimensions remain with size 1.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the reduction in backend-native kernel code.
- Use a numerically stable formulation: `max(x) + log(sum(exp(x - max(x))))`.
- Support positive and negative `dim` values.
- Support multi-axis reduction when `dim` is a list.
- For FP16/BF16 inputs, compute the reduction max, shifted exponentials, sum of exponentials, and final log in FP32.
- Do not exponentiate unshifted values; preserve the stable max-subtracted path for every reduced slice.
- Do not call `torch.logsumexp`, `torch.exp`, `torch.sum`, `torch.max`, or other PyTorch operators for the core computation in `ModelNew.forward`.
