# log_softmax

## Function

Computes log-softmax values along one dimension, equivalent to `torch.nn.functional.log_softmax`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the log-softmax computation in backend-native kernel code.
- Use a numerically stable formulation: subtract the maximum value along `dim` before computing log-sum-exp.
- Support positive and negative `dim` values.
- Only single-axis log-softmax is required for this operator.
- Do not call `torch.log_softmax`, `torch.nn.functional.log_softmax`, `torch.exp`, `torch.log`, `torch.sum`, `torch.max`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute max, exp-sum, and log-sum-exp in FP32.
- Use the stable formula `x - max - log(sum(exp(x - max)))`.
- Cast only the final log-probabilities back to the configured dtype.
