# multinomial

## Function

Deterministically samples indices from one or more multinomial distributions using explicit uniform random values.

This benchmark does not try to reproduce PyTorch RNG behavior. The random values are provided as an input tensor so the reference implementation and generated implementation can produce exactly comparable indices.

## Inputs

- `weights`: non-negative 1D or 2D tensor with shape from the selected benchmark shape.
- `uniforms`: FP32 tensor containing values in `[0, 1)`.
  - For 1D `weights`, shape is `[num_samples]`.
  - For 2D `weights`, shape is `[batch, num_samples]`.

The `weights` tensor uses the dtype selected by the shape entry.

## Output

- Int64 tensor of sampled indices.
- For 1D input, the output shape is `[num_samples]`.
- For 2D input, the output shape is `[batch, num_samples]`.

## Implementation Notes

- Implement multinomial sampling in backend-native kernel code.
- Support 1D and 2D input tensors.
- `num_samples` controls how many indices are sampled from each distribution.
- When `replacement` is false, each sampled index within one distribution must be unique.
- Treat the last dimension as the category dimension.
- Input weights are positive but not normalized.
- Use the provided `uniforms` tensor for sampling; do not generate random numbers inside `ModelNew.forward`.
- Accumulate weights in FP32 for the sampling CDF to reduce boundary differences for FP16/BF16 inputs.
- Do not call `torch.multinomial`, `torch.cumsum`, `torch.searchsorted`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Accumulate weights and CDF values in FP32 for FP16/BF16 inputs.
- Compare `uniforms` against the FP32 cumulative distribution to avoid boundary errors.
- For `replacement=False`, update temporary weights in FP32 after each selected index.
- Keep sampled index outputs exact in INT64.
