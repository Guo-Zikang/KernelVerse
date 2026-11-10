# topk

## Function

Selects the top-k values and indices along one dimension, equivalent to `torch.topk`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

All input tensors use the dtype selected by the shape entry.

## Output

- `values`: tensor containing the selected values.
- `indices`: int64 tensor containing the selected indices along `dim`.
- The size of the output along `dim` is `k`.

## Implementation Notes

- Implement both value selection and index tracking in backend-native kernel code.
- Support positive and negative `dim` values.
- Only single-axis top-k is required for this operator.
- Break ties deterministically by smaller original index first; for `largest=True` use `(value desc, index asc)`, and for `largest=False` use `(value asc, index asc)`.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compare scores in FP32 before top-k selection.
- Keep index outputs exact in INT64 and preserve deterministic tie-breaking.
- Gather selected values from the same precision path used by the reference, then cast values to the configured output dtype.
