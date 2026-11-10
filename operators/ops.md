# Operator Inventory

| Category | Count |
|---|---:|
| activation | 9 |
| attention | 53 |
| convolution | 7 |
| gemm | 8 |
| kv_cache | 13 |
| logits_sampling | 16 |
| moe | 19 |
| normalization | 6 |
| positional_embedding | 14 |
| quantization | 6 |
| reduction | 8 |
| state_space | 18 |
| tensor_ops | 23 |

## activation

- fatrelu_and_mul
- geglu
- gelu
- glu
- mul_and_silu
- sigmoid
- silu
- swiglu
- tanh

## attention

- alibi_attention
- append_attention
- batched_decode_attention
- batched_prefill_attention
- big_bird_attention
- block_local_attention
- block_sparse_attention
- cascade_attention
- causal_dilated_sliding_window_attention
- causal_multi_head_attention
- causal_sliding_window_attention
- cross_attention
- delta_net_attention
- disentangled_attention
- etc_global_local_attention
- flash_attention
- flash_attn_varlen
- flash_attn_with_kvcache
- flash_decoding_attention
- flash_mla_sparse
- flash_mla_with_kvcache
- gated_linear_attention
- grouped_query_attention
- kimi_delta_attention
- lightning_attn_diag_block
- lightning_attn_kv_reduce
- linear_attention
- linformer_attention
- locality_sensitive_hashing_attention
- longformer_attention
- merge_attn_states
- merge_state
- mla_decode
- mla_flash_attn_varlen_diff_headdims
- mla_prefill_context_chunk_attention
- mm_encoder_attention_sdpa
- mm_encoder_attention_varlen
- multi_head_attention
- multi_head_latent_attention
- multi_query_attention
- native_sparse_attention
- nystromformer_attention
- paged_attention_v1
- paged_attention_v2
- performer_attention
- ragged_kvcache_attention
- relative_position_bias_attention
- scaled_dot_product_attention
- self_attention
- streaming_llm_attention
- synthesizer_attention
- talking_heads_attention
- token_routing_sparse_attention

## convolution

- causal_conv1d_fwd
- causal_conv1d_update
- causal_conv2d
- conv2d
- conv3d
- depthwise_separable_conv1d
- glu_pointwise_conv

## gemm

- addmm
- baddbmm
- bmm
- int8_scaled_mm_azp
- linear
- lm_head_logits
- matmul
- matmul_v2

## kv_cache

- concat_and_cache_mla
- concat_and_cache_mla_rope
- kv_cache_write
- kv_layout_transform
- mla_kv_layout_transform
- paged_kv_cache_update
- reorg_kvcache
- sparse_k_cache_update
- static_sink_kv_populate
- transfer_kv_all_layer
- transfer_kv_per_layer
- transfer_mla_kv_all_layer
- transfer_mla_kv_per_layer

## logits_sampling

- apply_penalties
- apply_token_bitmask
- log_softmax
- logits_processor_scale_softcap
- masked_softmax
- multinomial
- paged_windowed_topk
- ragged_windowed_topk
- softmax
- sparse_decode_topk
- sparse_prefill_topk
- top_k_renorm_probs
- top_p_renorm_probs
- topk
- variable_length_topk
- windowed_topk

## moe

- apply_moe_activation
- apply_shuffle_mul_sum
- batched_moe_align_block_size
- biased_moe_topk_routing
- grouped_moe_mm
- grouped_topk_routing
- masked_silu_and_mul
- moe_align_block_size
- moe_fused_gate
- moe_lora_align_block_size
- moe_permute
- moe_sum_reduce
- moe_topk_routing
- moe_unpermute
- sigmoid_topk_routing
- softmax_topk_routing
- topk_weight_and_reduce_batched
- topk_weight_and_reduce_contiguous
- zero_expert_compute

## normalization

- add_rmsnorm
- l2_norm
- layer_norm
- mean_variance_norm
- rms_norm
- rmsnorm_gated

## positional_embedding

- deepseek_scaling_rotary_embedding
- dual_chunk_rotary_embedding
- dynamic_ntk_scaling_rotary_embedding
- gemma4_rotary_embedding
- linear_scaling_rotary_embedding
- llama3_rotary_embedding
- mrotary_embedding
- mrotary_embedding_interleaved
- ntk_scaling_rotary_embedding
- phi3_long_rope_scaled_rotary_embedding
- qk_norm_rope
- rotary_embedding
- xd_rotary_embedding
- yarn_scaling_rotary_embedding

## quantization

- dequantize
- dynamic_scaled_int8_quant
- per_token_group_quant_int8
- quantize_per_channel
- quantize_per_tensor
- static_scaled_int8_quant

## reduction

- argmax
- cumsum
- logsumexp
- max
- max_reduce
- mean_reduce
- sum_reduce
- var_mean

## state_space

- chunk_cumsum_dt
- chunk_fwd_o
- chunk_gated_delta_rule
- chunk_gated_delta_rule_fwd_h
- chunk_kda
- chunk_kda_scaled_dot_kkt_fwd
- chunk_local_cumsum
- kda_output_combine
- mamba_chunk_scan_combined_varlen
- recurrent_gated_delta_rule
- recurrent_gated_delta_rule_packed_decode
- selective_scan
- selective_state_update
- sigmoid_gating_delta_rule_update
- ssd_bmm
- ssd_chunk_scan
- ssd_chunk_state
- ssd_state_passing

## tensor_ops

- argsort
- cat
- chunk
- embedding
- gather
- gather_v2
- hadamard_transform
- index_add
- index_select
- masked_fill
- mul
- nonzero
- permute
- repeat_interleave
- scatter
- scatter_add
- scatter_reduce
- sort
- split
- stack
- take_along_dim
- transpose
- where
