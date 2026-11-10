"""Shape templates for kda_output_combine benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 4096, "hidden_dim": 4096, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 32768, "hidden_dim": 4096, "dtype": "FP32"},

        # hidden-dimension sweep
        {"num_tokens": 4096, "hidden_dim": 1024, "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 8192, "dtype": "FP32"},
        {"num_tokens": 2048, "hidden_dim": 16384, "dtype": "FP32"},

        # decode/batch-like sweep
        {"num_tokens": 1, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 32, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 256, "hidden_dim": 8192, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard projection width
        {"num_tokens": 4096, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "dtype": "FP16"},
        {"num_tokens": 4096, "hidden_dim": 4096, "dtype": "BF16"},

        # wider projection width
        {"num_tokens": 2048, "hidden_dim": 8192, "dtype": "FP32"},
        {"num_tokens": 2048, "hidden_dim": 8192, "dtype": "FP16"},
        {"num_tokens": 2048, "hidden_dim": 8192, "dtype": "BF16"},

        # decode-like small token count
        {"num_tokens": 128, "hidden_dim": 4096, "dtype": "FP32"},
        {"num_tokens": 128, "hidden_dim": 4096, "dtype": "FP16"},
        {"num_tokens": 128, "hidden_dim": 4096, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 512, 2048, 4096, 8192, 32768],
    "hidden_dim": [512, 1024, 2048, 4096, 8192, 11008, 16384],
    "dtype": ["FP32", "FP16", "BF16"],
}
