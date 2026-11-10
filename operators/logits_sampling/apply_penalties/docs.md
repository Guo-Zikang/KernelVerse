# apply_penalties

## Function

Applies presence, frequency, and repetition penalties to logits before sampling.

## Inputs

- `logits`: tensor with shape `[num_seqs, vocab_size]`.
- `prompt_tokens_tensor`: token ids with shape `[num_seqs, seq_len]`.
- `output_tokens_tensor`: token ids with shape `[num_seqs, seq_len]`.
- `presence_penalties`: tensor with shape `[num_seqs]`.
- `frequency_penalties`: tensor with shape `[num_seqs]`.
- `repetition_penalties`: tensor with shape `[num_seqs]`.

`logits` uses the dtype selected by the shape entry. Token tensors are `INT64`. Penalty tensors are `FP32`.

Token ids may include `vocab_size` as a sentinel value. Sentinel ids are ignored when applying penalties to logits.

## Output

Returns adjusted logits with shape `[num_seqs, vocab_size]` and the configured shape dtype.

## Implementation Notes

- Implement token counting, repetition penalty, frequency penalty, and presence penalty in backend-native kernel code.
- Repetition penalty applies to tokens seen in either prompt or output history.
- Frequency and presence penalties apply only to output history.
- Avoid unnecessary large temporary tensors when possible.
- Do not call `scatter_add_`, `torch.where`, or other PyTorch operators for the core computation in `ModelNew.forward`.

## Precision Notes

- Apply presence, frequency, and repetition penalties in FP32.
- Keep token counting and sentinel handling exact in integer logic.
- Cast only the final adjusted logits back to the configured dtype.
