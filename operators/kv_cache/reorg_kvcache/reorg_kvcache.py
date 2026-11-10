from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.kv_cache.reorg_kvcache.helpers import _resolve_dtype, validate_shape


class Model(nn.Module):
    """Reference distributed KV cache chunk reorganization."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)

    def forward(
        self,
        allgathered_kv_c_normed: torch.Tensor,
        allgathered_k_pe: torch.Tensor,
        padded_local_chunk_seq_lens: torch.Tensor,
        local_context_lens_allranks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_c_segments = []
        k_pe_segments = []
        total_tokens = int(padded_local_chunk_seq_lens.sum().item())
        src_token_idx = 0
        chunk_idx = int(local_context_lens_allranks[0, -2].item())
        chunk_size = int(local_context_lens_allranks[0, -1].item())

        for seq_idx, padded_len in enumerate(padded_local_chunk_seq_lens.tolist()):
            cur_context_lens = local_context_lens_allranks[seq_idx, :-2].tolist()
            for rank, local_context_len in enumerate(cur_context_lens):
                local_chunk_len = min(
                    max(0, int(local_context_len) - chunk_idx * chunk_size),
                    int(padded_len),
                )
                if local_chunk_len == 0:
                    continue
                start = rank * total_tokens + src_token_idx
                end = start + local_chunk_len
                kv_c_segments.append(allgathered_kv_c_normed[start:end])
                k_pe_segments.append(allgathered_k_pe[start:end])
            src_token_idx += int(padded_len)

        if not kv_c_segments:
            feature_dim = allgathered_kv_c_normed.shape[-1]
            empty_shape = (0, feature_dim)
            return (
                torch.empty(empty_shape, dtype=allgathered_kv_c_normed.dtype, device=allgathered_kv_c_normed.device),
                torch.empty(empty_shape, dtype=allgathered_k_pe.dtype, device=allgathered_k_pe.device),
            )

        return torch.cat(kv_c_segments, dim=0), torch.cat(k_pe_segments, dim=0)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    world_size = int(shape["world_size"])
    num_sequences = int(shape["num_sequences"])
    feature_dim = int(shape["feature_dim"])
    padded_chunk_len = int(shape["padded_chunk_len"])
    chunk_idx = int(shape["chunk_idx"])
    chunk_size = int(shape["chunk_size"])

    padded_local_chunk_seq_lens = torch.full(
        (num_sequences,),
        padded_chunk_len,
        dtype=torch.int32,
    )
    total_tokens = int(padded_local_chunk_seq_lens.sum().item())
    total_rows = world_size * total_tokens
    kv_c = torch.randn((total_rows, feature_dim), dtype=torch.float32).to(dtype)
    k_pe = torch.randn((total_rows, feature_dim), dtype=torch.float32).to(dtype)

    base_context = []
    min_valid_context = chunk_idx * chunk_size + 1
    for seq_idx in range(num_sequences):
        row = []
        for rank in range(world_size):
            valid_len = max(
                min_valid_context,
                chunk_idx * chunk_size + padded_chunk_len - seq_idx + rank,
            )
            row.append(valid_len)
        base_context.append(row + [chunk_idx, chunk_size])

    local_context_lens_allranks = torch.tensor(base_context, dtype=torch.int32)
    return kv_c, k_pe, padded_local_chunk_seq_lens, local_context_lens_allranks
