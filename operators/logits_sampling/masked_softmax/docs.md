# masked_softmax

## Function

Computes softmax probabilities along one dimension while excluding masked positions.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.
- `mask`: boolean-compatible tensor with the same shape as `tensor`; true entries are included in the softmax.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same shape as the input.
- Masked positions must be zero in the output.
- Unmasked values along `dim` should sum to approximately 1.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the masked softmax computation in backend-native kernel code.
- Use a numerically stable formulation over unmasked positions: subtract the maximum value along `dim` before exponentiation.
- Treat false mask entries as excluded from the softmax denominator.
- The input generator ensures at least one true mask entry along every reduction slice by setting index 0 along `dim` to true.
- Support positive and negative `dim` values.
- Only single-axis masked softmax is required for this operator.
- Do not call `torch.softmax`, `torch.nn.functional.softmax`, `masked_fill`, `torch.exp`, `torch.sum`, `torch.max`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the masked row maximum, exponentials, and denominator sum in FP32.
- Excluded mask positions must not contribute to the max or denominator.
- Cast only the final probabilities back to the configured dtype.
