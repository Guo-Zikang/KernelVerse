"""Shape templates for sigmoid_topk_routing benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_tokens": 128,
        "num_experts": 32,
        "topk": 2,
        "dtype": "FP32",
        "index_dtype": "INT32",
        "renormalize": True,
        "has_correction_bias": True,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 1024, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 8192, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},

        # expert-count sweep
        {"num_tokens": 2048, "num_experts": 32, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 2048, "num_experts": 256, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},

        # routing behavior sweep
        {"num_tokens": 4096, "num_experts": 256, "topk": 1, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 4096, "num_experts": 256, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False, "has_correction_bias": True},
        {"num_tokens": 4096, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": False},
    ],
    "dtype_sweep": [
        # correction-bias routing
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},

        # no-correction-bias routing
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": False},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": False},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": False},

        # index dtype
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT64", "renormalize": True, "has_correction_bias": True},
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "BF16", "index_dtype": "INT64", "renormalize": True, "has_correction_bias": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 32, 128, 256, 1024, 4096, 8192, 32768],
    "num_experts": [8, 16, 32, 64, 128, 256],
    "topk": [1, 2, 4, 8],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "renormalize": [True, False],
    "has_correction_bias": [True, False],
}
