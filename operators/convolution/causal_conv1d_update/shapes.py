"""Shape templates for causal_conv1d_update benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode-style single-token update
        {"batch": 1, "dim": 512, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 64, "dim": 128, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},

        # short chunk update
        {"batch": 8, "dim": 256, "seqlen": 4, "width": 3, "state_len": 8, "dtype": "FP32", "use_bias": True, "silu_activation": False},
        {"batch": 8, "dim": 512, "seqlen": 8, "width": 4, "state_len": 16, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 4, "dim": 1024, "seqlen": 16, "width": 4, "state_len": 32, "dtype": "FP32", "use_bias": False, "silu_activation": True},

        # state length sweep
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 2, "state_len": 1, "dtype": "FP32", "use_bias": True, "silu_activation": False},
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 3, "state_len": 8, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 64, "dtype": "FP32", "use_bias": True, "silu_activation": True},
    ],
    "dtype_sweep": [
        # decode-style update with SiLU
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP32", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "FP16", "use_bias": True, "silu_activation": True},
        {"batch": 16, "dim": 256, "seqlen": 1, "width": 4, "state_len": 3, "dtype": "BF16", "use_bias": True, "silu_activation": True},

        # short chunk update without activation
        {"batch": 8, "dim": 512, "seqlen": 8, "width": 3, "state_len": 16, "dtype": "FP32", "use_bias": True, "silu_activation": False},
        {"batch": 8, "dim": 512, "seqlen": 8, "width": 3, "state_len": 16, "dtype": "FP16", "use_bias": True, "silu_activation": False},
        {"batch": 8, "dim": 512, "seqlen": 8, "width": 3, "state_len": 16, "dtype": "BF16", "use_bias": True, "silu_activation": False},

        # bias-free path
        {"batch": 4, "dim": 1024, "seqlen": 16, "width": 4, "state_len": 32, "dtype": "FP32", "use_bias": False, "silu_activation": True},
        {"batch": 4, "dim": 1024, "seqlen": 16, "width": 4, "state_len": 32, "dtype": "FP16", "use_bias": False, "silu_activation": True},
        {"batch": 4, "dim": 1024, "seqlen": 16, "width": 4, "state_len": 32, "dtype": "BF16", "use_bias": False, "silu_activation": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch": [1, 2, 4, 8, 16, 32, 64],
    "dim": [64, 128, 256, 512, 1024, 2048],
    "seqlen": [1, 2, 4, 8, 16, 32],
    "width": [2, 3, 4],
    "state_len": [1, 2, 3, 4, 8, 16, 32, 64],
    "dtype": ["FP32", "FP16", "BF16"],
    "use_bias": [True, False],
    "silu_activation": [True, False],
}
