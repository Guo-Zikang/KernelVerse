"""Shape templates for per_token_group_quant_int8 benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 128, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 32768, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},

        # hidden dimension sweep
        {"num_tokens": 4096, "hidden_dim": 1024, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 8192, "group_size": 128, "dtype": "FP32", "eps": 1e-6},

        # group size sweep
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 32, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 64, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 256, "dtype": "FP32", "eps": 1e-6},

        # flattened batch * sequence layouts
        {"num_tokens": 32, "hidden_dim": 256, "group_size": 64, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 2048, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 8192, "hidden_dim": 8192, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
    ],
    "dtype_sweep": [
        # standard hidden layout
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "FP16", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 128, "dtype": "BF16", "eps": 1e-6},

        # smaller group size
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 64, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 64, "dtype": "FP16", "eps": 1e-6},
        {"num_tokens": 4096, "hidden_dim": 4096, "group_size": 64, "dtype": "BF16", "eps": 1e-6},

        # larger hidden dimension
        {"num_tokens": 2048, "hidden_dim": 8192, "group_size": 128, "dtype": "FP32", "eps": 1e-6},
        {"num_tokens": 2048, "hidden_dim": 8192, "group_size": 128, "dtype": "FP16", "eps": 1e-6},
        {"num_tokens": 2048, "hidden_dim": 8192, "group_size": 128, "dtype": "BF16", "eps": 1e-6},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [32, 128, 512, 2048, 4096, 8192, 32768],
    "hidden_dim": [256, 1024, 2048, 4096, 8192, 11008],
    "group_size": [32, 64, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
    "eps": [1e-6],
}
