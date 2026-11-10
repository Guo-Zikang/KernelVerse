# geglu

## Function

Applies the GEGLU activation. The input tensor is split into two equal chunks along `dim`, then computes `gelu(gate) * value`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

The selected `dim` must have an even size. All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same rank as the input.
- The output size along `dim` is half of the input size.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement GEGLU in backend-native kernel code as an elementwise gated activation after splitting along `dim`.
- Use the tanh approximation formula for GELU computation.
- Support non-last-dimension `dim` values when present in the selected shapes.
- Do not call `torch.nn.functional.gelu`, `torch.chunk`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the GELU gate with the tanh approximation in FP32 when practical, then multiply by the value chunk before final cast.
