from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.apply_token_bitmask.helpers import (
    BITS_PER_BLOCK,
    UINT32_MASK,
    _resolve_dtype,
    validate_shape,
)


class Model(nn.Module):
    """Masks disallowed token logits using a packed token bitmask."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)

    def forward(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        out = logits.clone()
        rows = indices.long() if indices.numel() > 0 else torch.arange(out.size(0), device=out.device)
        vocab_size = out.size(-1)
        token_ids = torch.arange(vocab_size, device=out.device)
        word_ids = token_ids // BITS_PER_BLOCK
        bit_ids = token_ids % BITS_PER_BLOCK

        for row in rows:
            row_int = int(row.item())
            words = bitmask[row_int].to(torch.int64) & UINT32_MASK
            allowed = ((words[word_ids] >> bit_ids) & 1).bool()
            out[row_int, ~allowed] = -torch.inf
        return out


def _pack_bitmask(allowed: torch.Tensor) -> torch.Tensor:
    batch, vocab_size = allowed.shape
    bitmask = torch.zeros((batch, vocab_size // BITS_PER_BLOCK), dtype=torch.int64)
    for bit in range(BITS_PER_BLOCK):
        bitmask |= allowed[:, bit::BITS_PER_BLOCK].to(torch.int64) << bit
    bitmask = torch.where(bitmask >= 2 ** 31, bitmask - 2 ** 32, bitmask)
    return bitmask.to(torch.int32)


def generate_inputs(shape: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    dtype = _resolve_dtype(shape["dtype"])
    batch = int(shape["batch"])
    vocab_size = int(shape["vocab_size"])
    logits = torch.randn((batch, vocab_size), dtype=torch.float32).to(dtype)
    allowed = torch.rand((batch, vocab_size)) > 0.35
    bitmask = _pack_bitmask(allowed)
    if shape["use_indices"]:
        indices = torch.randperm(batch, dtype=torch.int32)[: max(1, batch // 2)]
    else:
        indices = torch.empty((0,), dtype=torch.int32)
    return logits, bitmask, indices
