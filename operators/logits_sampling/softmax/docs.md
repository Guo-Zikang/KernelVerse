# softmax

## Function

Computes softmax probabilities along one dimension, equivalent to `torch.nn.functional.softmax`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match PyTorch softmax behavior for the configured input dtype.
- Values along `dim` should sum to approximately 1.

## Implementation Notes

- Implement the softmax computation in backend-native kernel code.
- Use a numerically stable formulation: subtract the maximum value along `dim` before exponentiation.
- Support positive and negative `dim` values.
- Only single-axis softmax is required for this operator.
- Do not call `torch.softmax`, `torch.nn.functional.softmax`, `torch.exp`, `torch.sum`, `torch.max`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the row maximum, exponentials, and denominator sum in FP32.
- Use the stable path `exp(x - max)` and divide in FP32.
- Cast only the final probabilities back to the configured dtype.
