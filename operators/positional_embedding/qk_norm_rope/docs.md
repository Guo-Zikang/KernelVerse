# qk_norm_rope

## Function

Splits a packed QKV tensor, applies RMSNorm to Q and K per head, applies rotary embedding to Q and K, then packs Q, K, and V back together.

## Inputs

- `qkv`: packed QKV tensor with shape `[num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]`.
- `q_weight`: RMSNorm weight for Q with shape `[head_dim]`.
- `k_weight`: RMSNorm weight for K with shape `[head_dim]`.
- `position_ids`: int32 position tensor with shape `[num_tokens]`.

All floating input tensors use the dtype selected by the shape entry.

## Parameters

- `num_heads_q`, `num_heads_k`, `num_heads_v`, `head_dim`, `rotary_dim`, `eps`, and `is_neox` are known from the shape entry.

## Output

- Packed QKV tensor with the same shape as `qkv`.
- Output dtype must match the configured shape dtype.

## Implementation Notes

- Implement Q/K RMSNorm, rotary embedding, and QKV repacking in backend-native kernel code.
- Support both NeoX and interleaved rotary layouts.
- Do not call `torch.rsqrt`, `torch.mean`, `torch.pow`, `torch.sin`, `torch.cos`, RMSNorm libraries, RoPE libraries, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- For FP16/BF16, correctness is evaluated against the benchmark's native low-precision reference path, not an FP32 oracle; preserve the exact RMSNorm, frequency, angle, sin/cos, rotation, and repacking path.
- Preserve the exact RMSNorm epsilon handling and NeoX/interleaved rotary layout; large errors usually come from normalization scale, phase, or layout mismatches.
