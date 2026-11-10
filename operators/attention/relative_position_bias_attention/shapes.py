"""Shape templates for relative_position_bias_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 2,
        "seq_len": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "num_buckets": 32,
        "max_distance": 256,
        "bias": True,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 2, "seq_len": 128, "num_heads": 8, "head_dim": 64, "num_buckets": 32, "max_distance": 128, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_buckets": 64, "max_distance": 512, "bias": True, "dtype": "FP32"},

        # bucket count and max-distance sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 16, "max_distance": 128, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 64, "max_distance": 512, "bias": True, "dtype": "FP32"},

        # head count and head dimension sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 8, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 512, "num_heads": 32, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 128, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 256, "num_heads": 16, "head_dim": 256, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},

        # bias mode and batch sweep
        {"batch_size": 2, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "BF16"},

        # larger batch
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 512, "num_heads": 16, "head_dim": 64, "num_buckets": 32, "max_distance": 256, "bias": True, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_buckets": 64, "max_distance": 512, "bias": False, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_buckets": 64, "max_distance": 512, "bias": False, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "num_buckets": 64, "max_distance": 512, "bias": False, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "seq_len": [128, 256, 512, 1024, 2048],
    "num_heads": [8, 16, 32, 40, 64],
    "head_dim": [64, 96, 128, 256],
    "num_buckets": [16, 32, 64],
    "max_distance": [128, 256, 512, 1024],
    "bias": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
