# embedding

## Function

Performs embedding lookup from a weight table using integer token ids.

## Inputs

- `input_ids`: int tensor with shape from `input_shape`.
- `weight`: embedding table with shape `[num_embeddings, embedding_dim]`.

## Output

- Tensor with shape `input_shape + [embedding_dim]`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- For every element `input_ids[...]`, copy `weight[input_ids[...], :]` to the corresponding output row.
- `input_ids` are generated in range `[0, num_embeddings)`.
- This is a gather-style memory bandwidth operator over the embedding dimension.
- Copy selected embedding values exactly without arithmetic, normalization, or dtype conversion.
- For FP16/BF16 weights, preserve the original stored values in the output rows.
- Do not call PyTorch embedding, gather, index_select, advanced indexing, or other PyTorch operators for the core computation in `ModelNew.forward`.
