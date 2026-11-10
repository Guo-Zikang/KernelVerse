from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from operators.logits_sampling.apply_penalties.helpers import (
    _resolve_dtype,
    validate_shape,
)


def _count_token_bins(
    tokens: torch.Tensor,
    num_seqs: int,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_counts = torch.zeros((num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    return bin_counts, bin_counts > 0


class Model(nn.Module):
    """Applies presence, frequency, and repetition penalties to logits."""

    def __init__(self, shape: Dict[str, Any]) -> None:
        super().__init__()
        validate_shape(shape)
        self.output_dtype = _resolve_dtype(shape["dtype"])

    def forward(
        self,
        logits: torch.Tensor,
        prompt_tokens_tensor: torch.Tensor,
        output_tokens_tensor: torch.Tensor,
        presence_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        repetition_penalties: torch.Tensor,
    ) -> torch.Tensor:
        num_seqs, vocab_size = logits.shape
        _, prompt_mask = _count_token_bins(prompt_tokens_tensor, num_seqs, vocab_size)
        output_bin_counts, output_mask = _count_token_bins(output_tokens_tensor, num_seqs, vocab_size)

        out = logits.float().clone()
        repeated = prompt_mask | output_mask
        pos_mask = out > 0
        repetition = repetition_penalties.float().unsqueeze(1)
        out = torch.where(repeated & pos_mask, out / repetition, out)
        out = torch.where(repeated & ~pos_mask, out * repetition, out)

        frequency = frequency_penalties.float().unsqueeze(1)
        presence = presence_penalties.float().unsqueeze(1)
        out -= frequency * output_bin_counts.float()
        out -= presence * output_mask.float()
        return out.to(self.output_dtype)


def generate_inputs(
    shape: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_shape(shape)
    num_seqs = int(shape["num_seqs"])
    vocab_size = int(shape["vocab_size"])
    seq_len = int(shape["seq_len"])
    dtype = _resolve_dtype(shape["dtype"])

    logits = torch.randn((num_seqs, vocab_size), dtype=torch.float32).to(dtype)
    prompt_tokens_tensor = torch.randint(0, vocab_size + 1, (num_seqs, seq_len), dtype=torch.int64)
    output_tokens_tensor = torch.randint(0, vocab_size + 1, (num_seqs, seq_len), dtype=torch.int64)
    presence_penalties = torch.rand((num_seqs,), dtype=torch.float32)
    frequency_penalties = torch.rand((num_seqs,), dtype=torch.float32)
    repetition_penalties = torch.rand((num_seqs,), dtype=torch.float32) + 1.0
    return (
        logits,
        prompt_tokens_tensor,
        output_tokens_tensor,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
