"""Shape templates for apply_moe_activation benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token-count sweep
        {"num_tokens": 128, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 32768, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},

        # hidden-dimension sweep
        {"num_tokens": 4096, "hidden_dim": 1024, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 8192, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 2048, "hidden_dim": 16384, "activation": "silu", "dtype": "FP32"},

        # decode/batch-like sweep
        {"num_tokens": 1, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 32, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 256, "hidden_dim": 8192, "activation": "silu", "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard gated activation
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "FP16"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "BF16"},

        # gelu gated activation
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "gelu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "gelu", "dtype": "FP16"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "gelu", "dtype": "BF16"},

        # no-mul activation
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu_no_mul", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu_no_mul", "dtype": "FP16"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu_no_mul", "dtype": "BF16"},
    ],
    "activation_sweep": [
        # gated activation variants
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "gelu", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "relu2", "dtype": "FP32"},

        # MoE-specific gated activation variants
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "swigluoai", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "swiglustep", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "silu_no_mul", "dtype": "FP32"},

        # no-mul activation variants
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "gelu_no_mul", "dtype": "FP32"},
        {"num_tokens": 4096, "hidden_dim": 4096, "activation": "relu2_no_mul", "dtype": "FP32"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 512, 2048, 4096, 8192, 32768],
    "hidden_dim": [512, 1024, 2048, 4096, 8192, 11008, 16384],
    "activation": ["silu", "gelu", "relu2", "swigluoai", "swiglustep", "silu_no_mul", "gelu_no_mul", "relu2_no_mul"],
    "dtype": ["FP32", "FP16", "BF16"],
}
