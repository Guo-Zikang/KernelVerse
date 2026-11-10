# layer_norm

## Function

Computes layer normalization over the trailing `normalized_shape` dimensions.

## Inputs

- `tensor`: input tensor with shape `shape`.
- `weight`: scale tensor with shape `normalized_shape`.
- `bias`: bias tensor with shape `normalized_shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as `tensor`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Normalize over the trailing dimensions listed in `normalized_shape`.
- Use `eps` from the shape entry when computing inverse standard deviation.
- Implement reduction, scale, and bias in backend-native kernel code.
- Do not call `torch.nn.functional.layer_norm`, `torch.var`, `torch.mean`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Compute mean and variance over `normalized_shape` in FP32, especially for FP16/BF16 inputs.
- Apply `eps`, inverse standard deviation, weight, and bias in FP32.
- Cast only the final output back to the configured dtype.
