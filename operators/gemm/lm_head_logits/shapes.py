"""Shape templates for lm_head_logits benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode and small batch token counts
        {"num_tokens": 1, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 16, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},

        # prefill token counts
        {"num_tokens": 1024, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 4096, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 8192, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},

        # hidden dimension sweep
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 8192, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 11008, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},

        # vocabulary size sweep
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 49152, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 128256, "dtype": "FP32", "use_bias": True},

        # bias behavior
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 128, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": False},
        {"num_tokens": 1024, "hidden_dim": 8192, "vocab_size": 49152, "dtype": "FP32", "use_bias": False},
    ],
    "dtype_sweep": [
        # decode-style projection
        {"num_tokens": 16, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 16, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP16", "use_bias": True},
        {"num_tokens": 16, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "BF16", "use_bias": True},

        # prefill projection
        {"num_tokens": 1024, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP32", "use_bias": True},
        {"num_tokens": 1024, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "FP16", "use_bias": True},
        {"num_tokens": 1024, "hidden_dim": 4096, "vocab_size": 32000, "dtype": "BF16", "use_bias": True},

        # large vocabulary projection without bias
        {"num_tokens": 128, "hidden_dim": 8192, "vocab_size": 128256, "dtype": "FP32", "use_bias": False},
        {"num_tokens": 128, "hidden_dim": 8192, "vocab_size": 128256, "dtype": "FP16", "use_bias": False},
        {"num_tokens": 128, "hidden_dim": 8192, "vocab_size": 128256, "dtype": "BF16", "use_bias": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 16, 128, 256, 512, 1024],
    "hidden_dim": [4096, 8192],
    "vocab_size": [32000, 49152],
    "dtype": ["FP32", "FP16", "BF16"],
    "use_bias": [True, False],
}
