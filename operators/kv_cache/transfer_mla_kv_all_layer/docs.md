# transfer_mla_kv_all_layer

## Function

Copies selected compressed MLA KV-cache rows for every transformer layer from source cache slots to destination cache slots.

## Inputs

- `src_layers`: source compressed MLA cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `dst_layers`: destination compressed MLA cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `src_indices`: source row indices with shape `[num_items]`.
- `dst_indices`: destination row indices with shape `[num_items]`.

`src_layers` and `dst_layers` use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns one copied cache tensor with shape `[num_layers, num_pages, item_elems]`.

## Implementation Notes

- This is a memory-bound all-layer MLA KV-cache relocation operator.
- For each layer and each `i`, copy `src_layers[layer, src_indices[i]]` to `out[layer, dst_indices[i]]`.
- Preserve destination rows that are not listed in `dst_indices`.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache relocation operator; copy source rows to destination rows without changing values.
- Preserve source compressed MLA cache values exactly in the configured dtype for every layer.
- Keep source/destination index handling exact and preserve destination rows that are not listed.
