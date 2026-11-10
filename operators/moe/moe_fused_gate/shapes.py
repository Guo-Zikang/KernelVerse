"""Shape templates for moe_fused_gate benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 5000, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 1024, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 8192, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 32768, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},

        # expert scale sweep
        {"num_tokens": 4096, "num_experts": 128, "num_expert_group": 4, "topk_group": 2, "topk": 4, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 512, "num_expert_group": 16, "topk_group": 8, "topk": 16, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},

        # fused shared expert sweep
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 9, "num_fused_shared_experts": 1, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 10, "num_fused_shared_experts": 2, "dtype": "FP32", "index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # DeepSeek-style grouped gate
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "BF16", "index_dtype": "INT32"},

        # fused shared experts
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 10, "num_fused_shared_experts": 2, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 10, "num_fused_shared_experts": 2, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 10, "num_fused_shared_experts": 2, "dtype": "BF16", "index_dtype": "INT32"},

        # index dtype
        {"num_tokens": 4096, "num_experts": 128, "num_expert_group": 4, "topk_group": 2, "topk": 4, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "num_experts": 128, "num_expert_group": 4, "topk_group": 2, "topk": 4, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT64"},
        {"num_tokens": 8192, "num_experts": 256, "num_expert_group": 8, "topk_group": 4, "topk": 8, "num_fused_shared_experts": 0, "dtype": "FP32", "index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [128, 1024, 4096, 8192, 32768],
    "num_experts": [64, 128, 256, 512],
    "num_expert_group": [4, 8, 16],
    "topk_group": [1, 2, 4, 8],
    "topk": [4, 8, 9, 10, 16],
    "num_fused_shared_experts": [0, 1, 2],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
}
