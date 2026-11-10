# argmax

## Function

Computes maximum-value indices along one dimension, equivalent to `torch.argmax(tensor, dim=...)`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- `indices`: int64 tensor containing the index of the selected maximum value along `dim`.
- If `keepdim` is true, the reduced dimension remains with size 1.

## Implementation Notes

- Implement argmax reduction and index tracking in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis reduction is required for this operator.
- Preserve comparison semantics exactly, including negative values and NaN behavior if present.
- Match PyTorch tie behavior as closely as practical: if multiple equal max values occur along the reduction dimension, return the first index.
- Do not call `torch.argmax`, `torch.max`, `torch.amax`, or other PyTorch operators for the core computation in `ModelNew.forward`.
