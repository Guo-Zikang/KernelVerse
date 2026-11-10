"""Shape templates for selective_state_update benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 1, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 512, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},

        # head and channel sweep
        {"num_tokens": 64, "num_heads": 4, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 16, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 32, "head_dim": 128, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},

        # state size sweep
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 8, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 32, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 16, "head_dim": 128, "dstate": 64, "dtype": "FP32", "dt_softplus": True, "has_z": True},

        # option sweep
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": False, "has_z": True},
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": False},
        {"num_tokens": 256, "num_heads": 16, "head_dim": 64, "dstate": 32, "dtype": "FP32", "dt_softplus": False, "has_z": False},
    ],
    "dtype_sweep": [
        # standard state update
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "FP16", "dt_softplus": True, "has_z": True},
        {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "dstate": 16, "dtype": "BF16", "dt_softplus": True, "has_z": True},

        # no gate
        {"num_tokens": 128, "num_heads": 16, "head_dim": 64, "dstate": 16, "dtype": "FP32", "dt_softplus": True, "has_z": False},
        {"num_tokens": 128, "num_heads": 16, "head_dim": 64, "dstate": 16, "dtype": "FP16", "dt_softplus": True, "has_z": False},
        {"num_tokens": 128, "num_heads": 16, "head_dim": 64, "dstate": 16, "dtype": "BF16", "dt_softplus": True, "has_z": False},

        # larger recurrent state
        {"num_tokens": 32, "num_heads": 16, "head_dim": 128, "dstate": 32, "dtype": "FP32", "dt_softplus": False, "has_z": True},
        {"num_tokens": 32, "num_heads": 16, "head_dim": 128, "dstate": 32, "dtype": "FP16", "dt_softplus": False, "has_z": True},
        {"num_tokens": 32, "num_heads": 16, "head_dim": 128, "dstate": 32, "dtype": "BF16", "dt_softplus": False, "has_z": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 16, 64, 128, 256, 512],
    "num_heads": [1, 4, 8, 16, 32],
    "head_dim": [32, 64, 128],
    "dstate": [8, 16, 32, 64],
    "dtype": ["FP32", "FP16", "BF16"],
    "dt_softplus": [True, False],
    "has_z": [True, False],
}
