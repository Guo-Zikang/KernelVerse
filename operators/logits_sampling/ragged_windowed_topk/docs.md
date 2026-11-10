# ragged_windowed_topk

## Function

Selects top-k score positions from each row within a valid window, then shifts the selected positions into a ragged global index space.

## Inputs

- `score`: input score tensor with shape `[batch, length]`.
- `lengths`: int32 tensor with shape `[batch]`.
- `topk_indices_offset`: int32 tensor with shape `[batch]`.
- `row_starts`: int32 tensor with shape `[batch]`.

All floating input tensors use the dtype selected by the shape entry.
Each row only considers `score[row, row_starts[row]:row_starts[row] + lengths[row]]` as valid.

## Output

- `indices`: int32 tensor with shape `[batch, topk]`.

## Implementation Notes

- Select local top-k positions inside each valid window.
- Add `row_starts[row]` to convert local top-k positions into absolute row positions.
- Add `topk_indices_offset[row]` to shift absolute row positions into the ragged global index space.
- Only row-wise top-k over contiguous windows is required for this operator.
- The generated `lengths[row]` values are at least `topk`.
- Break ties deterministically by smaller local window index first before adding offsets.
- Do not call `torch.topk`, `torch.sort`, `torch.argsort`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compare scores in FP32 before top-k selection.
- Keep shifted index outputs exact in INT32 and preserve deterministic tie-breaking.
- Do not let low-precision rounding change selected positions before adding offsets.
