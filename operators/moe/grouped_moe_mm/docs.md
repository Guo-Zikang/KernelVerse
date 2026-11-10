# grouped_moe_mm

## Function

Runs grouped MoE expert matrix multiplications. Each expert consumes a contiguous token slice from `a_tensors`, multiplies it by that expert's weight matrix, and concatenates all expert outputs.

Optional activation and weight scales are applied before the matrix multiplication.

## Inputs

- `a_tensors`: routed token tensor with shape `[total_tokens, K]`.
- `b_tensors`: expert weight tensor with shape `[num_experts, K, N]`.
- `a_scales`: activation scale tensor. Shape is `[total_tokens]` when `per_act_token` is true, otherwise `[1]`.
- `b_scales`: weight scale tensor. Shape is `[num_experts, N]` when `per_out_ch` is true, otherwise `[num_experts]`.
- `expert_offsets`: prefix offsets with shape `[num_experts + 1]`; expert `i` consumes `a_tensors[expert_offsets[i]:expert_offsets[i + 1]]`.

Floating inputs use the dtype selected by the shape entry. `expert_offsets` is INT32.

## Output

Returns one tensor with shape `[total_tokens, N]`.

The output uses the dtype selected by the shape entry.

## Implementation Notes

- Implement the grouped expert GEMM, scale application, and output concatenation in backend-native kernel code.
- Do not call cuBLAS, CUTLASS, vendor GEMM libraries, `torch.matmul`, `torch.mm`, `torch.cat`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16 inputs, accumulate the K dimension in FP32 or hardware mixed-precision accumulators.
- Apply activation scales and weight scales in FP32 before or during accumulation.
- Cast only the final output back to the configured dtype.
