# index_add

## Function

Adds values from a source tensor into an input tensor along one dimension, equivalent to `torch.index_add`.

## Inputs

- `tensor`: input tensor with shape `input_shape`.
- `index`: int64 1D tensor with shape `[index_size]`; each value selects the destination position along `dim`.
- `source`: tensor with the same rank as `tensor`; its shape equals `input_shape` except dimension `dim` is `index_size`.

All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with shape `input_shape`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement single-axis indexed accumulation in backend-native kernel code.
- Support positive and negative `dim` values.
- Apply `alpha` exactly as `output[index[i]] += alpha * source[i]` along the selected dimension.
- Every index value is in `[0, input_shape[dim])`.
- Repeated index values are allowed and must accumulate all corresponding source values.
- For FP16/BF16 inputs, accumulate repeated-index contributions in FP32, then cast once to the output dtype.
- Preserve repeated-index semantics exactly; all contributions to the same destination must participate in the accumulation.
- Do not call `torch.index_add`, `torch.scatter_add`, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
