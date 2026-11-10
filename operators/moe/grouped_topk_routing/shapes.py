"""Shape templates for grouped_topk_routing benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 128, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 16, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 128, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 1024, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},

        # group-count sweep
        {"num_tokens": 256, "num_experts": 32, "topk": 4, "num_expert_group": 4, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 128, "topk": 4, "num_expert_group": 16, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 256, "topk": 8, "num_expert_group": 16, "topk_group": 4, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},

        # routing behavior sweep
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": False, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "has_correction_bias": True},
        {"num_tokens": 256, "num_experts": 64, "topk": 8, "num_expert_group": 8, "topk_group": 4, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT64", "renormalize": True, "scoring_func": "sigmoid", "routed_scaling_factor": 1.0, "has_correction_bias": True},
    ],
    "dtype_sweep": [
        # softmax grouped routing
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP16", "weight_dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "BF16", "weight_dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},

        # sigmoid grouped routing with correction bias
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "has_correction_bias": True},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "FP16", "weight_dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "has_correction_bias": True},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "num_expert_group": 8, "topk_group": 2, "input_dtype": "BF16", "weight_dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "has_correction_bias": True},

        # index dtype
        {"num_tokens": 256, "num_experts": 128, "topk": 8, "num_expert_group": 16, "topk_group": 4, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
        {"num_tokens": 256, "num_experts": 128, "topk": 8, "num_expert_group": 16, "topk_group": 4, "input_dtype": "FP32", "weight_dtype": "FP32", "index_dtype": "INT64", "renormalize": True, "scoring_func": "softmax", "routed_scaling_factor": 1.0, "has_correction_bias": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 32, 128, 256, 1024, 4096],
    "num_experts": [8, 16, 32, 64, 128, 256],
    "topk": [1, 2, 4, 8],
    "num_expert_group": [1, 2, 4, 8, 16],
    "topk_group": [1, 2, 4, 8],
    "input_dtype": ["FP32", "FP16", "BF16"],
    "weight_dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "renormalize": [True, False],
    "scoring_func": ["softmax", "sigmoid"],
    "routed_scaling_factor": [1.0, 2.5],
    "has_correction_bias": [True, False],
}
