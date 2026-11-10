# max_reduce

## Function

Computes the maximum value of a tensor along one or more dimensions, equivalent to `torch.amax`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor reduced along `dim`.
- If `keepdim` is true, reduced dimensions remain with size 1.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement the reduction in backend-native kernel code.
- Support positive and negative `dim` values.
- Support multi-axis reduction when `dim` is a list.
- Preserve comparison semantics exactly, including negative values and NaN behavior if present.
- Do not call `torch.amax`, `torch.max`, or other PyTorch operators for the core computation in `ModelNew.forward`.
