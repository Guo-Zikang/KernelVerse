# sort

## Function

Sorts tensor values and returns sorted values and original indices along one dimension, equivalent to `torch.sort`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- `values`: tensor containing sorted values.
- `indices`: int64 tensor containing the original indices of the sorted values along `dim`.
- Output tensors have the same shape as the input tensor.

## Implementation Notes

- Implement both value sorting and index tracking in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis sort is required for this operator.
- Support `descending=True` and `descending=False`.
- Always use stable sorting: equal values must preserve their original index order.
- Preserve comparison and ordering semantics exactly, including ties, signed values, stable ordering, and NaN behavior if present.
- Index outputs must be exact INT64 original indices; do not approximate, synthesize, or infer indices from shape metadata.
- Do not call `torch.sort`, `torch.argsort`, `torch.topk`, or other PyTorch operators for the core computation in `ModelNew.forward`.
