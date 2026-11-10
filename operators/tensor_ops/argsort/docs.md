# argsort

## Function

Returns the indices that sort tensor values along one dimension, equivalent to `torch.argsort`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- `indices`: int64 tensor containing the original indices of the sorted values along `dim`.
- Output tensor has the same shape as the input tensor.

## Implementation Notes

- Implement index sorting in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis argsort is required for this operator.
- Always use stable sorting: equal values must preserve their original index order.
- Preserve comparison and ordering semantics exactly, including `descending`, signed values, and NaN behavior if present.
- Index outputs must be exact INT64 original indices; do not approximate, synthesize, or infer indices from shape metadata.
- Do not call `torch.argsort`, `torch.sort`, `torch.topk`, or other PyTorch operators for the core computation in `ModelNew.forward`.
