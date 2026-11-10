"""Shape templates for topk_weight_and_reduce_contiguous benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_tokens": 256,
        "topk": 4,
        "hidden_dim": 1024,
        "dtype": "FP32",
        "index_dtype": "INT32",
        "apply_router_weight_on_input": False,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 16, "topk": 4, "hidden_dim": 1024, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 256, "topk": 4, "hidden_dim": 1024, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 4096, "topk": 4, "hidden_dim": 1024, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},

        # hidden-size sweep
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 256, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 8192, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},

        # routing layout sweep
        {"num_tokens": 1024, "topk": 1, "hidden_dim": 4096, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 4096, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": True},
        {"num_tokens": 1024, "topk": 8, "hidden_dim": 4096, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
    ],
    "dtype_sweep": [
        # weighted reduce
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "FP16", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "BF16", "index_dtype": "INT32", "apply_router_weight_on_input": False},

        # pre-weighted input reduce
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": True},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "FP16", "index_dtype": "INT32", "apply_router_weight_on_input": True},
        {"num_tokens": 1024, "topk": 4, "hidden_dim": 2048, "dtype": "BF16", "index_dtype": "INT32", "apply_router_weight_on_input": True},

        # index dtype
        {"num_tokens": 512, "topk": 8, "hidden_dim": 4096, "dtype": "FP32", "index_dtype": "INT32", "apply_router_weight_on_input": False},
        {"num_tokens": 512, "topk": 8, "hidden_dim": 4096, "dtype": "FP32", "index_dtype": "INT64", "apply_router_weight_on_input": False},
        {"num_tokens": 512, "topk": 8, "hidden_dim": 4096, "dtype": "BF16", "index_dtype": "INT64", "apply_router_weight_on_input": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 128, 256, 1024, 4096, 8192],
    "topk": [1, 2, 4, 8],
    "hidden_dim": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "apply_router_weight_on_input": [True, False],
}
