"""Shape templates for biased MoE top-k routing benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_tokens": 128,
        "num_experts": 64,
        "topk": 4,
        "dtype": "FP32",
        "index_dtype": "INT32",
        "renormalize": True,
        "scoring_func": "softmax",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 16, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 128, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 1024, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},

        # expert-count sweep
        {"num_tokens": 256, "num_experts": 32, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},

        # routing behavior sweep
        {"num_tokens": 256, "num_experts": 64, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 64, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid"},
    ],
    "dtype_sweep": [
        # softmax routing
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},

        # sigmoid routing
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid"},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid"},
        {"num_tokens": 256, "num_experts": 64, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": True, "scoring_func": "sigmoid"},

        # index dtype
        {"num_tokens": 256, "num_experts": 128, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True, "scoring_func": "softmax"},
        {"num_tokens": 256, "num_experts": 128, "topk": 8, "dtype": "FP32", "index_dtype": "INT64", "renormalize": True, "scoring_func": "softmax"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 32, 128, 256, 1024, 4096],
    "num_experts": [8, 16, 32, 64, 128, 256],
    "topk": [1, 2, 4, 8],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "renormalize": [True, False],
    "scoring_func": ["softmax", "sigmoid"],
}
