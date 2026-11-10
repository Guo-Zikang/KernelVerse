# repeat_interleave

## Function

Repeats tensor elements, equivalent to `torch.repeat_interleave`.

## Inputs

- `tensor`: input tensor with shape `shape`.

All input tensors use the dtype selected by the shape entry.

## Output

- If `dim` is `None`, the output is a flattened 1D tensor with `numel(tensor) * repeats` elements.
- If `dim` is specified, the output shape matches `shape` except that dimension is multiplied by `repeats`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement repeat-interleave in backend-native kernel code.
- Support `dim=None`, positive `dim`, and negative `dim` values.
- `repeats` is a scalar integer for all elements in the current shape contract.
- This is a data movement operator: copy repeated input values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 inputs, preserve the original stored values in each repeated output element.
- Do not call `torch.repeat_interleave`, `repeat`, `expand`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
