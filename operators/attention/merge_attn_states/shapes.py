"""Shape templates for merge_attn_states benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 1, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 2048, "num_heads": 16, "head_size": 128, "dtype": "FP32"},

        # head count sweep
        {"num_tokens": 128, "num_heads": 8, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 32, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 64, "head_size": 128, "dtype": "FP32"},

        # head size sweep
        {"num_tokens": 128, "num_heads": 16, "head_size": 64, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 16, "head_size": 256, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard merge
        {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "FP16"},
        {"num_tokens": 128, "num_heads": 16, "head_size": 128, "dtype": "BF16"},

        # prefill-like token count
        {"num_tokens": 2048, "num_heads": 16, "head_size": 128, "dtype": "FP32"},
        {"num_tokens": 2048, "num_heads": 16, "head_size": 128, "dtype": "FP16"},
        {"num_tokens": 2048, "num_heads": 16, "head_size": 128, "dtype": "BF16"},

        # wider head
        {"num_tokens": 128, "num_heads": 32, "head_size": 256, "dtype": "FP32"},
        {"num_tokens": 128, "num_heads": 32, "head_size": 256, "dtype": "FP16"},
        {"num_tokens": 128, "num_heads": 32, "head_size": 256, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 512, 1024, 2048, 4096],
    "num_heads": [4, 8, 16, 32, 64],
    "head_size": [32, 64, 96, 128, 192, 256],
    "dtype": ["FP32", "FP16", "BF16"],
}
