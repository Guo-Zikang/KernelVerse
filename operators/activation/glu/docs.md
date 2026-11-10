# glu

## Function

Applies the gated linear unit activation. The input tensor is split into two equal chunks along `dim`, then computes `gate * sigmoid(value)`.

## Inputs

- `tensor`: input tensor with shape from the selected benchmark shape.

The selected `dim` must have an even size. All input tensors use the dtype selected by the shape entry.

## Output

- Tensor with the same rank as the input.
- The output size along `dim` is half of the input size.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement GLU in backend-native kernel code as an elementwise gated activation after splitting along `dim`.
- Support non-last-dimension `dim` values when present in the selected shapes.
- Do not call `torch.nn.functional.glu`, `torch.sigmoid`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, compute the sigmoid gate and gated multiply in FP32 when practical; preserve the same split order as the reference.
