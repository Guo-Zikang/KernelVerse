# transfer_kv_per_layer

## Function

Copies selected K/V cache rows for one transformer layer from source cache slots to destination cache slots.

## Inputs

- `src_k`: source key cache tensor with shape `[num_pages, item_elems]`.
- `dst_k`: destination key cache tensor with shape `[num_pages, item_elems]`.
- `src_v`: source value cache tensor with shape `[num_pages, item_elems]`.
- `dst_v`: destination value cache tensor with shape `[num_pages, item_elems]`.
- `src_indices`: source row indices with shape `[num_items]`.
- `dst_indices`: destination row indices with shape `[num_items]`.

`src_k`, `dst_k`, `src_v`, and `dst_v` use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns a tuple:

- `out_k`: copied key cache tensor with shape `[num_pages, item_elems]`.
- `out_v`: copied value cache tensor with shape `[num_pages, item_elems]`.

## Implementation Notes

- This is a memory-bound KV cache relocation operator.
- For each `i`, copy `src_k[src_indices[i]]` to `out_k[dst_indices[i]]` and copy `src_v[src_indices[i]]` to `out_v[dst_indices[i]]`.
- Preserve destination rows that are not listed in `dst_indices`.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache relocation operator; copy source rows to destination rows without changing values.
- Preserve source key/value values exactly in the configured dtype.
- Keep source/destination index handling exact and preserve destination rows that are not listed.
