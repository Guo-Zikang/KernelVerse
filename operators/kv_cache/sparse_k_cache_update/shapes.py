"""Shape templates for sparse_k_cache_update benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 128, "num_slots": 4096, "head_dim": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode token count sweep
        {"num_tokens": 1, "num_slots": 2048, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 64, "num_slots": 4096, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 512, "num_slots": 8192, "head_dim": 128, "dtype": "FP32"},

        # cache size sweep
        {"num_tokens": 128, "num_slots": 1024, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 8192, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 32000, "head_dim": 128, "dtype": "FP32"},

        # head dimension sweep
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 64, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 256, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard decode cache update
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 128, "dtype": "FP16"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 128, "dtype": "BF16"},

        # many token updates
        {"num_tokens": 512, "num_slots": 8192, "head_dim": 128, "dtype": "FP32"},
        {"num_tokens": 512, "num_slots": 8192, "head_dim": 128, "dtype": "FP16"},
        {"num_tokens": 512, "num_slots": 8192, "head_dim": 128, "dtype": "BF16"},

        # wider key dimension
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 256, "dtype": "FP32"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 256, "dtype": "FP16"},
        {"num_tokens": 128, "num_slots": 4096, "head_dim": 256, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 64, 128, 256, 512, 1024],
    "num_slots": [64, 256, 1024, 2048, 4096, 8192, 32000],
    "head_dim": [64, 80, 96, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
}
