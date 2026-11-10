"""Shape templates for moe_unpermute benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 512, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 8192, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},

        # hidden dimension sweep
        {"num_tokens": 2048, "hidden_dim": 1024, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 8192, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},

        # routing fanout sweep
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 1, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 4, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "topk": 8, "num_experts": 128, "dtype": "FP32", "index_dtype": "INT32"},
    ],
    "dtype_sweep": [
        # standard MoE unpermutation
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 4096, "topk": 2, "num_experts": 64, "dtype": "BF16", "index_dtype": "INT32"},

        # larger routing fanout
        {"num_tokens": 4096, "hidden_dim": 4096, "topk": 4, "num_experts": 128, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "topk": 4, "num_experts": 128, "dtype": "FP16", "index_dtype": "INT32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "topk": 4, "num_experts": 128, "dtype": "BF16", "index_dtype": "INT32"},

        # index dtype
        {"num_tokens": 2048, "hidden_dim": 2048, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT32"},
        {"num_tokens": 2048, "hidden_dim": 2048, "topk": 2, "num_experts": 64, "dtype": "FP32", "index_dtype": "INT64"},
        {"num_tokens": 4096, "hidden_dim": 4096, "topk": 4, "num_experts": 128, "dtype": "FP32", "index_dtype": "INT64"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [128, 512, 1024, 2048, 4096, 8192, 16384],
    "hidden_dim": [512, 1024, 2048, 4096, 8192, 11008],
    "topk": [1, 2, 4, 8],
    "num_experts": [16, 32, 64, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
}
