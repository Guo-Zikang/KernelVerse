"""Shape templates for recurrent_gated_delta_rule_packed_decode benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch size sweep
        {"batch_size": 1, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 64, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},

        # head-count sweep
        {"batch_size": 16, "num_heads": 4, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 16, "num_heads": 16, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 8, "num_heads": 32, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},

        # key/value dimension sweep
        {"batch_size": 16, "num_heads": 8, "key_dim": 32, "value_dim": 32, "dtype": "FP32"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 128, "dtype": "FP32"},
        {"batch_size": 8, "num_heads": 8, "key_dim": 128, "value_dim": 128, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard packed decode
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP16"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "BF16"},

        # larger batch decode
        {"batch_size": 64, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP32"},
        {"batch_size": 64, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "FP16"},
        {"batch_size": 64, "num_heads": 8, "key_dim": 64, "value_dim": 64, "dtype": "BF16"},

        # wider value dimension
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 128, "dtype": "FP32"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 128, "dtype": "FP16"},
        {"batch_size": 16, "num_heads": 8, "key_dim": 64, "value_dim": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
    "num_heads": [1, 4, 8, 16, 32],
    "key_dim": [32, 64, 128],
    "value_dim": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
