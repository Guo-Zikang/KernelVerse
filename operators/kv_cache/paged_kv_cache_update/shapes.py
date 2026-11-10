"""Shape templates for paged_kv_cache_update benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "num_tokens": 256,
        "num_blocks": 1024,
        "block_size": 16,
        "num_heads": 8,
        "head_dim": 128,
        "dtype": "FP16",
        "index_dtype": "INT64",
        "has_invalid_slots": False,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode-style slot updates
        {"num_tokens": 1, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 32, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 256, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},

        # prefill-style slot updates
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 2048, "num_blocks": 2048, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 8192, "num_blocks": 4096, "block_size": 16, "num_heads": 16, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},

        # layout and invalid slot behavior
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 8, "num_heads": 8, "head_dim": 64, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": True},
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": True},
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 32, "num_heads": 16, "head_dim": 256, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": True},
    ],
    "dtype_sweep": [
        # decode-style updates
        {"num_tokens": 256, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 256, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP16", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 256, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "BF16", "index_dtype": "INT64", "has_invalid_slots": False},

        # invalid slot updates
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": True},
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "FP16", "index_dtype": "INT64", "has_invalid_slots": True},
        {"num_tokens": 512, "num_blocks": 1024, "block_size": 16, "num_heads": 8, "head_dim": 128, "dtype": "BF16", "index_dtype": "INT64", "has_invalid_slots": True},

        # index dtype
        {"num_tokens": 1024, "num_blocks": 2048, "block_size": 16, "num_heads": 16, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT32", "has_invalid_slots": False},
        {"num_tokens": 1024, "num_blocks": 2048, "block_size": 16, "num_heads": 16, "head_dim": 128, "dtype": "FP32", "index_dtype": "INT64", "has_invalid_slots": False},
        {"num_tokens": 1024, "num_blocks": 2048, "block_size": 16, "num_heads": 16, "head_dim": 128, "dtype": "BF16", "index_dtype": "INT64", "has_invalid_slots": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192],
    "num_blocks": [16, 64, 256, 512, 1024, 2048, 4096],
    "block_size": [1, 4, 8, 16, 32],
    "num_heads": [1, 4, 8, 16, 32],
    "head_dim": [64, 80, 96, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "has_invalid_slots": [True, False],
}
