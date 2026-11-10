# mul

## Function

Performs elementwise multiplication with PyTorch-style broadcasting.

## Inputs

- `lhs`: tensor with shape `lhs_dims`.
- `rhs`: tensor with shape `rhs_dims`.

`lhs_dims` and `rhs_dims` must be broadcast-compatible.

## Output

Returns one tensor with the broadcasted output shape and dtype `dtype`.

## Implementation Notes

- Support broadcasting by aligning dimensions from the trailing axis.
- Compute `lhs * rhs` for every output element.
- Implement broadcasting index calculation and multiplication in backend-native kernel code.
- For FP16/BF16 inputs, match the benchmark reference dtype behavior; avoid unnecessary dtype conversions.
- Cast to the output dtype only once when writing the final output.
- Do not call PyTorch `mul`, broadcasting helpers, TensorIterator, or other PyTorch operators for the core computation in `ModelNew.forward`.
