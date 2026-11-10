# variable_length_topk

## Function

Selects top-k indices from each score row while respecting a per-row valid length.

## Inputs

- `score`: input score tensor with shape `[batch, length]`.
- `lengths`: int32 tensor with shape `[batch]`.

All floating input tensors use the dtype selected by the shape entry.
Each row only considers `score[row, :lengths[row]]` as valid.
Values in `score[row, lengths[row]:]` are invalid padding and must be ignored even if they are numerically large.

## Output

- `indices`: int32 tensor with shape `[batch, topk]`.

## Implementation Notes

- For rows where `lengths[row] > topk`, return the top-k indices within the valid prefix.
- For rows where `lengths[row] <= topk`, return indices `[0, 1, ..., lengths[row] - 1]` and fill the remaining output positions with `-1`.
- Tie-breaking is deterministic: when two valid scores are equal, the smaller index ranks first.
- The backend comparator should order candidates by `(score desc, index asc)`.
- The output contains indices only, not top-k values.
- Only row-wise top-k over the last dimension is required for this operator.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compare scores in FP32 before top-k selection.
- Keep index outputs exact in INT32 and preserve deterministic tie-breaking.
- Do not let low-precision rounding change selected indices.
