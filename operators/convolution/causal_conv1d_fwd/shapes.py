"""Shape templates for causal_conv1d_fwd benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch": 4, "dim": 128, "seqlen": 512, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 512, "seqlen": 4096, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},

        # convolution width and activation sweep
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 2, "dtype": "FP32", "use_bias": True, "silu_activation": False},
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "FP32", "use_bias": False, "silu_activation": True},

        # batch and channel sweep
        {"batch": 1, "dim": 1024, "seqlen": 1024, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 128, "seqlen": 4096, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 32, "dim": 256, "seqlen": 1024, "width": 3, "dtype": "FP32", "use_bias": True, "silu_activation": False},
    ],
    "dtype_sweep": [
        # standard causal conv with SiLU
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "FP16", "use_bias": True, "silu_activation": True},
        {"batch": 8, "dim": 256, "seqlen": 2048, "width": 4, "dtype": "BF16", "use_bias": True, "silu_activation": True},

        # no activation path
        {"batch": 16, "dim": 128, "seqlen": 4096, "width": 3, "dtype": "FP32", "use_bias": True, "silu_activation": False},
        {"batch": 16, "dim": 128, "seqlen": 4096, "width": 3, "dtype": "FP16", "use_bias": True, "silu_activation": False},
        {"batch": 16, "dim": 128, "seqlen": 4096, "width": 3, "dtype": "BF16", "use_bias": True, "silu_activation": False},

        # bias-free path
        {"batch": 4, "dim": 512, "seqlen": 1024, "width": 4, "dtype": "FP32", "use_bias": False, "silu_activation": True},
        {"batch": 4, "dim": 512, "seqlen": 1024, "width": 4, "dtype": "FP16", "use_bias": False, "silu_activation": True},
        {"batch": 4, "dim": 512, "seqlen": 1024, "width": 4, "dtype": "BF16", "use_bias": False, "silu_activation": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch": [1, 2, 4, 8, 16, 32],
    "dim": [64, 128, 256, 512, 1024, 2048],
    "seqlen": [128, 256, 512, 1024, 2048, 4096, 8192],
    "width": [2, 3, 4],
    "dtype": ["FP32", "FP16", "BF16"],
    "use_bias": [True, False],
    "silu_activation": [True, False],
}
