"""Operator category metadata.

This module records semantic labels for benchmark operators. Operator loading
is handled by operators.registry; this file describes how operators should be
grouped for analysis and reporting.
"""

OPERATOR_CATEGORIES = {
    "matmul": {
        "type": "gemm",
        "category": "gemm",
    },
    "matmul_v2": {
        "type": "gemm",
        "category": "gemm",
    },
    "addmm": {
        "type": "gemm",
        "category": "gemm",
    },
    "bmm": {
        "type": "gemm",
        "category": "gemm",
    },
    "baddbmm": {
        "type": "gemm",
        "category": "gemm",
    },
    "linear": {
        "type": "gemm",
        "category": "gemm",
    },
    "lm_head_logits": {
        "type": "gemm",
        "category": "gemm",
    },
    "int8_scaled_mm_azp": {
        "type": "gemm",
        "category": "gemm",
    },
    "add_rmsnorm": {
        "type": "normalization",
        "category": "normalization",
    },
    "rmsnorm_gated": {
        "type": "normalization",
        "category": "normalization",
    },
    "qk_norm_rope": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "kda_output_combine": {
        "type": "state_space",
        "category": "state_space",
    },
    "concat_and_cache_mla_rope": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "append_attention": {
        "type": "attention",
        "category": "attention",
    },
    "concat_and_cache_mla": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "kv_cache_write": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "paged_kv_cache_update": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "sparse_k_cache_update": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "static_sink_kv_populate": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "reorg_kvcache": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "transfer_kv_per_layer": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "transfer_kv_all_layer": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "transfer_mla_kv_per_layer": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "transfer_mla_kv_all_layer": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "kv_layout_transform": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "mla_kv_layout_transform": {
        "type": "kv_cache",
        "category": "kv_cache",
    },
    "sigmoid_gating_delta_rule_update": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_gated_delta_rule": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_gated_delta_rule_fwd_h": {
        "type": "state_space",
        "category": "state_space",
    },
    "recurrent_gated_delta_rule": {
        "type": "state_space",
        "category": "state_space",
    },
    "recurrent_gated_delta_rule_packed_decode": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_kda": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_kda_scaled_dot_kkt_fwd": {
        "type": "state_space",
        "category": "state_space",
    },
    "mamba_chunk_scan_combined_varlen": {
        "type": "state_space",
        "category": "state_space",
    },
    "selective_scan": {
        "type": "state_space",
        "category": "state_space",
    },
    "selective_state_update": {
        "type": "state_space",
        "category": "state_space",
    },
    "ssd_bmm": {
        "type": "state_space",
        "category": "state_space",
    },
    "ssd_chunk_state": {
        "type": "state_space",
        "category": "state_space",
    },
    "ssd_chunk_scan": {
        "type": "state_space",
        "category": "state_space",
    },
    "ssd_state_passing": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_fwd_o": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_local_cumsum": {
        "type": "state_space",
        "category": "state_space",
    },
    "chunk_cumsum_dt": {
        "type": "state_space",
        "category": "state_space",
    },
    "lightning_attn_diag_block": {
        "type": "attention",
        "category": "attention",
    },
    "batched_decode_attention": {
        "type": "attention",
        "category": "attention",
    },
    "batched_prefill_attention": {
        "type": "attention",
        "category": "attention",
    },
    "ragged_kvcache_attention": {
        "type": "attention",
        "category": "attention",
    },
    "cascade_attention": {
        "type": "attention",
        "category": "attention",
    },
    "lightning_attn_kv_reduce": {
        "type": "attention",
        "category": "attention",
    },
    "merge_attn_states": {
        "type": "attention",
        "category": "attention",
    },
    "merge_state": {
        "type": "attention",
        "category": "attention",
    },
    "mla_decode": {
        "type": "attention",
        "category": "attention",
    },
    "mla_flash_attn_varlen_diff_headdims": {
        "type": "attention",
        "category": "attention",
    },
    "mla_prefill_context_chunk_attention": {
        "type": "attention",
        "category": "attention",
    },
    "mm_encoder_attention_sdpa": {
        "type": "attention",
        "category": "attention",
    },
    "mm_encoder_attention_varlen": {
        "type": "attention",
        "category": "attention",
    },
    "conv2d": {
        "type": "convolution",
        "category": "convolution",
    },
    "conv3d": {
        "type": "convolution",
        "category": "convolution",
    },
    "causal_conv2d": {
        "type": "convolution",
        "category": "convolution",
    },
    "causal_conv1d_fwd": {
        "type": "convolution",
        "category": "convolution",
    },
    "causal_conv1d_update": {
        "type": "convolution",
        "category": "convolution",
    },
    "depthwise_separable_conv1d": {
        "type": "convolution",
        "category": "convolution",
    },
    "glu_pointwise_conv": {
        "type": "convolution",
        "category": "convolution",
    },
    "embedding": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "gather_v2": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "gather": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "take_along_dim": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "index_select": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "index_add": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "scatter": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "scatter_add": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "scatter_reduce": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "cat": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "stack": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "repeat_interleave": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "permute": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "split": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "chunk": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "transpose": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "hadamard_transform": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "deepseek_scaling_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "dual_chunk_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "dynamic_ntk_scaling_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "ntk_scaling_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "gemma4_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "linear_scaling_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "llama3_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "mrotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "mrotary_embedding_interleaved": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "phi3_long_rope_scaled_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "xd_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "yarn_scaling_rotary_embedding": {
        "type": "positional_embedding",
        "category": "positional_embedding",
    },
    "quantize_per_tensor": {
        "type": "quantization",
        "category": "quantization",
    },
    "quantize_per_channel": {
        "type": "quantization",
        "category": "quantization",
    },
    "dequantize": {
        "type": "quantization",
        "category": "quantization",
    },
    "dynamic_scaled_int8_quant": {
        "type": "quantization",
        "category": "quantization",
    },
    "static_scaled_int8_quant": {
        "type": "quantization",
        "category": "quantization",
    },
    "per_token_group_quant_int8": {
        "type": "quantization",
        "category": "quantization",
    },
    "masked_fill": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "mul": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "nonzero": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "where": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "masked_softmax": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "softmax": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "log_softmax": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "top_k_renorm_probs": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "top_p_renorm_probs": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "variable_length_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "windowed_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "paged_windowed_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "ragged_windowed_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "sparse_decode_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "sparse_prefill_topk": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "multinomial": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "apply_penalties": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "apply_token_bitmask": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "logits_processor_scale_softcap": {
        "type": "logits_sampling",
        "category": "logits_sampling",
    },
    "moe_topk_routing": {
        "type": "moe",
        "category": "moe",
    },
    "biased_moe_topk_routing": {
        "type": "moe",
        "category": "moe",
    },
    "grouped_topk_routing": {
        "type": "moe",
        "category": "moe",
    },
    "sigmoid_topk_routing": {
        "type": "moe",
        "category": "moe",
    },
    "softmax_topk_routing": {
        "type": "moe",
        "category": "moe",
    },
    "apply_moe_activation": {
        "type": "moe",
        "category": "moe",
    },
    "masked_silu_and_mul": {
        "type": "moe",
        "category": "moe",
    },
    "apply_shuffle_mul_sum": {
        "type": "moe",
        "category": "moe",
    },
    "batched_moe_align_block_size": {
        "type": "moe",
        "category": "moe",
    },
    "moe_align_block_size": {
        "type": "moe",
        "category": "moe",
    },
    "moe_lora_align_block_size": {
        "type": "moe",
        "category": "moe",
    },
    "moe_permute": {
        "type": "moe",
        "category": "moe",
    },
    "moe_unpermute": {
        "type": "moe",
        "category": "moe",
    },
    "moe_sum_reduce": {
        "type": "moe",
        "category": "moe",
    },
    "zero_expert_compute": {
        "type": "moe",
        "category": "moe",
    },
    "moe_fused_gate": {
        "type": "moe",
        "category": "moe",
    },
    "grouped_moe_mm": {
        "type": "moe",
        "category": "moe",
    },
    "topk_weight_and_reduce_contiguous": {
        "type": "moe",
        "category": "moe",
    },
    "topk_weight_and_reduce_batched": {
        "type": "moe",
        "category": "moe",
    },
    "sort": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "argsort": {
        "type": "tensor_ops",
        "category": "tensor_ops",
    },
    "argmax": {
        "type": "reduction",
        "category": "reduction",
    },
    "max": {
        "type": "reduction",
        "category": "reduction",
    },
    "max_reduce": {
        "type": "reduction",
        "category": "reduction",
    },
    "sum_reduce": {
        "type": "reduction",
        "category": "reduction",
    },
    "mean_reduce": {
        "type": "reduction",
        "category": "reduction",
    },
    "var_mean": {
        "type": "reduction",
        "category": "reduction",
    },
    "logsumexp": {
        "type": "reduction",
        "category": "reduction",
    },
    "glu": {
        "type": "activation",
        "category": "activation",
    },
    "swiglu": {
        "type": "activation",
        "category": "activation",
    },
    "geglu": {
        "type": "activation",
        "category": "activation",
    },
    "fatrelu_and_mul": {
        "type": "activation",
        "category": "activation",
    },
    "mul_and_silu": {
        "type": "activation",
        "category": "activation",
    },
    "cumsum": {
        "type": "reduction",
        "category": "reduction",
    },
    "layer_norm": {
        "type": "core_compute",
        "category": "level1_basic",
    },
    "rms_norm": {
        "type": "normalization",
        "category": "normalization",
    },
    "mean_variance_norm": {
        "type": "normalization",
        "category": "normalization",
    },
    "l2_norm": {
        "type": "normalization",
        "category": "normalization",
    },
    "gelu": {
        "type": "activation",
        "category": "activation",
    },
    "silu": {
        "type": "activation",
        "category": "activation",
    },
    "sigmoid": {
        "type": "activation",
        "category": "activation",
    },
    "tanh": {
        "type": "activation",
        "category": "activation",
    },
    "scaled_dot_product_attention": {
        "type": "attention",
        "category": "attention",
    },
    "multi_head_attention": {
        "type": "core_compute",
        "category": "level3_attention",
    },
    "causal_multi_head_attention": {
        "type": "attention",
        "category": "attention",
    },
    "alibi_attention": {
        "type": "attention",
        "category": "attention",
    },
    "relative_position_bias_attention": {
        "type": "attention",
        "category": "attention",
    },
    "talking_heads_attention": {
        "type": "attention",
        "category": "attention",
    },
    "synthesizer_attention": {
        "type": "attention",
        "category": "attention",
    },
    "disentangled_attention": {
        "type": "attention",
        "category": "attention",
    },
    "multi_head_latent_attention": {
        "type": "attention",
        "category": "attention",
    },
    "self_attention": {
        "type": "attention",
        "category": "attention",
    },
    "cross_attention": {
        "type": "attention",
        "category": "attention",
    },
    "flash_attention": {
        "type": "attention",
        "category": "attention",
    },
    "flash_attn_varlen": {
        "type": "attention",
        "category": "attention",
    },
    "flash_attn_with_kvcache": {
        "type": "attention",
        "category": "attention",
    },
    "flash_mla_sparse": {
        "type": "attention",
        "category": "attention",
    },
    "flash_mla_with_kvcache": {
        "type": "attention",
        "category": "attention",
    },
    "flash_decoding_attention": {
        "type": "attention",
        "category": "attention",
    },
    "grouped_query_attention": {
        "type": "attention",
        "category": "attention",
    },
    "multi_query_attention": {
        "type": "attention",
        "category": "attention",
    },
    "linear_attention": {
        "type": "attention",
        "category": "attention",
    },
    "gated_linear_attention": {
        "type": "attention",
        "category": "attention",
    },
    "delta_net_attention": {
        "type": "attention",
        "category": "attention",
    },
    "kimi_delta_attention": {
        "type": "attention",
        "category": "attention",
    },
    "linformer_attention": {
        "type": "attention",
        "category": "attention",
    },
    "nystromformer_attention": {
        "type": "attention",
        "category": "attention",
    },
    "performer_attention": {
        "type": "attention",
        "category": "attention",
    },
    "block_local_attention": {
        "type": "attention",
        "category": "attention",
    },
    "block_sparse_attention": {
        "type": "attention",
        "category": "attention",
    },
    "native_sparse_attention": {
        "type": "attention",
        "category": "attention",
    },
    "causal_sliding_window_attention": {
        "type": "attention",
        "category": "attention",
    },
    "causal_dilated_sliding_window_attention": {
        "type": "attention",
        "category": "attention",
    },
    "streaming_llm_attention": {
        "type": "attention",
        "category": "attention",
    },
    "longformer_attention": {
        "type": "attention",
        "category": "attention",
    },
    "etc_global_local_attention": {
        "type": "attention",
        "category": "attention",
    },
    "big_bird_attention": {
        "type": "attention",
        "category": "attention",
    },
    "locality_sensitive_hashing_attention": {
        "type": "attention",
        "category": "attention",
    },
    "token_routing_sparse_attention": {
        "type": "attention",
        "category": "attention",
    },
    "paged_attention_v1": {
        "type": "attention",
        "category": "attention",
    },
    "paged_attention_v2": {
        "type": "attention",
        "category": "attention",
    },
}
