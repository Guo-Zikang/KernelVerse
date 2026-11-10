# transfer_kv_all_layer

## Function

Copies selected K/V cache rows for every transformer layer from source cache slots to destination cache slots.

## Inputs

- `src_k_layers`: source key cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `dst_k_layers`: destination key cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `src_v_layers`: source value cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `dst_v_layers`: destination value cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `src_indices`: source row indices with shape `[num_items]`.
- `dst_indices`: destination row indices with shape `[num_items]`.

`src_k_layers`, `dst_k_layers`, `src_v_layers`, and `dst_v_layers` use the dtype selected by the shape entry. Indices use `index_dtype`.

## Output

Returns a tuple:

- `out_k`: copied key cache tensor with shape `[num_layers, num_pages, item_elems]`.
- `out_v`: copied value cache tensor with shape `[num_layers, num_pages, item_elems]`.

## Implementation Notes

- This is a memory-bound all-layer KV cache relocation operator.
- For each layer and each `i`, copy `src_k_layers[layer, src_indices[i]]` to `out_k[layer, dst_indices[i]]` and copy `src_v_layers[layer, src_indices[i]]` to `out_v[layer, dst_indices[i]]`.
- Preserve destination rows that are not listed in `dst_indices`.
- Do not call PyTorch indexing, scatter, gather, clone, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- This is a cache relocation operator; copy source rows to destination rows without changing values.
- Preserve source key/value values exactly in the configured dtype for every layer.
- Keep source/destination index handling exact and preserve destination rows that are not listed.
