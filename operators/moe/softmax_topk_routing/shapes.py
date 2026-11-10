"""Shape templates for softmax_topk_routing benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_tokens": 128,
        "num_experts": 32,
        "topk": 2,
        "dtype": "FP32",
        "index_dtype": "INT32",
        "renormalize": False,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 1024, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 8192, "num_experts": 32, "topk": 2, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},

        # expert-count sweep
        {"num_tokens": 2048, "num_experts": 32, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 2048, "num_experts": 512, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},

        # routing behavior sweep
        {"num_tokens": 4096, "num_experts": 256, "topk": 1, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 4096, "num_experts": 256, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True},
        {"num_tokens": 4096, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
    ],
    "dtype_sweep": [
        # default routing
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 2048, "num_experts": 128, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": False},

        # renormalized routing
        {"num_tokens": 2048, "num_experts": 256, "topk": 4, "dtype": "FP32", "index_dtype": "INT32", "renormalize": True},
        {"num_tokens": 2048, "num_experts": 256, "topk": 4, "dtype": "FP16", "index_dtype": "INT32", "renormalize": True},
        {"num_tokens": 2048, "num_experts": 256, "topk": 4, "dtype": "BF16", "index_dtype": "INT32", "renormalize": True},

        # index dtype
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT32", "renormalize": False},
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "FP32", "index_dtype": "INT64", "renormalize": False},
        {"num_tokens": 1024, "num_experts": 256, "topk": 8, "dtype": "BF16", "index_dtype": "INT64", "renormalize": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 32, 128, 256, 1024, 4096, 8192, 32768],
    "num_experts": [8, 16, 32, 64, 128, 256, 512],
    "topk": [1, 2, 4, 8],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "renormalize": [True, False],
}
