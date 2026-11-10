from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.attention.cascade_attention.helpers import (
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Reference cascade decode attention with shared prefix and unique suffix."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.num_q_heads = int(shape["num_q_heads"])
        self.num_kv_heads = int(shape["num_kv_heads"])
        self.head_dim = int(shape["head_dim"])
        self.output_dtype = _resolve_dtype(shape["dtype"])
        self.scale = float(shape.get("softmax_scale", self.head_dim ** -0.5))
        baseline_compute_mode = shape.get("_baseline_compute_mode", "performance")
        self.compute_dtype = (
            torch.float32
            if baseline_compute_mode == "correctness"
            else self.output_dtype
        )

    def forward(
        self,
        q: torch.Tensor,
        shared_k: torch.Tensor,
        shared_v: torch.Tensor,
        unique_k: torch.Tensor,
        unique_v: torch.Tensor,
        unique_indptr: torch.Tensor,
    ) -> torch.Tensor:
        q_compute = q.to(self.compute_dtype)
        shared_k_compute = shared_k.to(self.compute_dtype)
        shared_v_compute = shared_v.to(self.compute_dtype)
        unique_k_compute = unique_k.to(self.compute_dtype)
        unique_v_compute = unique_v.to(self.compute_dtype)

        q_per_kv = self.num_q_heads // self.num_kv_heads
        kv_head_indices = torch.arange(self.num_q_heads, device=q.device) // q_per_kv
        shared_key = shared_k_compute[:, kv_head_indices, :]
        shared_value = shared_v_compute[:, kv_head_indices, :]

        outputs = []
        for batch_idx in range(q.shape[0]):
            shared_out, shared_lse = self._attention_partial(
                q_compute[batch_idx],
                shared_key,
                shared_value,
            )

            unique_start = int(unique_indptr[batch_idx].item())
            unique_end = int(unique_indptr[batch_idx + 1].item())
            if unique_end > unique_start:
                unique_key = unique_k_compute[unique_start:unique_end, kv_head_indices, :]
                unique_value = unique_v_compute[unique_start:unique_end, kv_head_indices, :]
                unique_out, unique_lse = self._attention_partial(
                    q_compute[batch_idx],
                    unique_key,
                    unique_value,
                )
            else:
                unique_out = torch.zeros_like(shared_out)
                unique_lse = torch.full_like(shared_lse, float("-inf"))

            outputs.append(self._merge_partials(shared_out, shared_lse, unique_out, unique_lse))

        return torch.stack(outputs, dim=0).to(self.output_dtype)

    def _attention_partial(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = (key * query.unsqueeze(0)).sum(dim=-1).transpose(0, 1)
        scores = scores * self.scale
        lse = torch.logsumexp(scores.to(torch.float32), dim=-1).to(self.compute_dtype)
        probs = torch.softmax(scores, dim=-1)
        output = (probs.unsqueeze(-1) * value.permute(1, 0, 2)).sum(dim=1)
        return output, lse

    @staticmethod
    def _merge_partials(
        shared_output: torch.Tensor,
        shared_lse: torch.Tensor,
        unique_output: torch.Tensor,
        unique_lse: torch.Tensor,
    ) -> torch.Tensor:
        max_lse = torch.maximum(shared_lse, unique_lse)
        shared_weight = torch.exp(shared_lse - max_lse)
        unique_weight = torch.exp(unique_lse - max_lse)
        denom = shared_weight + unique_weight
        return (
            shared_output * (shared_weight / denom).unsqueeze(-1)
            + unique_output * (unique_weight / denom).unsqueeze(-1)
        )


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    batch_size = int(shape["batch_size"])
    shared_len = int(shape["shared_len"])
    unique_lens = [int(seq_len) for seq_len in shape["unique_lens"]]
    total_unique_tokens = sum(unique_lens)
    num_q_heads = int(shape["num_q_heads"])
    num_kv_heads = int(shape["num_kv_heads"])
    head_dim = int(shape["head_dim"])
    dtype = _resolve_dtype(shape["dtype"])

    q = torch.randn((batch_size, num_q_heads, head_dim), dtype=torch.float32).to(dtype)
    shared_k = torch.randn((shared_len, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    shared_v = torch.randn((shared_len, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    unique_k = torch.randn((total_unique_tokens, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    unique_v = torch.randn((total_unique_tokens, num_kv_heads, head_dim), dtype=torch.float32).to(dtype)
    unique_indptr = _make_indptr(unique_lens)
    return q, shared_k, shared_v, unique_k, unique_v, unique_indptr


def _make_indptr(seq_lens: list[int]) -> torch.Tensor:
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + int(seq_len))
    return torch.tensor(offsets, dtype=torch.int32)
