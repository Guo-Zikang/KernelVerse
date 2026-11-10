# mean_variance_norm

## Function

Applies stored mean and inverse-standard-deviation normalization over the last dimension:

```text
output = (input - global_mean) * global_invstd
```

This operator does not compute mean or variance from the input at runtime.

## Inputs

- `input_`: tensor with shape from the selected shape entry.

The input tensor uses the dtype selected by the shape entry.

## Parameters

- `global_mean`: parameter with shape `[input_size]`, where `input_size` is the last dimension of `input_`.
- `global_invstd`: parameter with shape `[input_size]`.

## Output

- Tensor with the same shape as `input_`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- `ModelNew.__init__` must create `global_mean` first and `global_invstd` second, matching the reference model parameter order.
- Broadcast `global_mean` and `global_invstd` across all leading dimensions.
- Implement the subtract, multiply, and writeback in backend-native kernel code.
- Do not call PyTorch normalization, subtraction, multiplication, broadcasting helpers, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Convert input, `global_mean`, and `global_invstd` to FP32 for `(input - global_mean) * global_invstd`.
- Broadcast parameters exactly across leading dimensions.
- Cast only the final output back to the configured dtype.
