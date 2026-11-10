"""Shape templates for apply_shuffle_mul_sum benchmark cases."""

SINGLE_SHAPES = {
    "default": {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"m": 128, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 32768, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},

        # top-k sweep
        {"m": 4096, "topk": 1, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 2, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 8, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},

        # hidden-size sweep
        {"m": 4096, "topk": 4, "hidden_size": 1024, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 8192, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 2048, "topk": 4, "hidden_size": 16384, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
    ],
    "dtype_sweep": [
        # weighted expert merge
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP16", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "BF16", "index_dtype": "INT32", "has_factors": True},

        # unweighted expert merge
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": False},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP16", "index_dtype": "INT32", "has_factors": False},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "BF16", "index_dtype": "INT32", "has_factors": False},

        # index dtype
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT32", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT64", "has_factors": True},
        {"m": 4096, "topk": 4, "hidden_size": 4096, "dtype": "FP32", "index_dtype": "INT64", "has_factors": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "m": [1, 8, 32, 128, 512, 2048, 4096, 8192, 32768],
    "topk": [1, 2, 4, 8],
    "hidden_size": [512, 1024, 2048, 4096, 8192, 11008, 16384],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "has_factors": [True, False],
}
