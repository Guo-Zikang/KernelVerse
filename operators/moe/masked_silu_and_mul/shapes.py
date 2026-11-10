"""Shape templates for masked_silu_and_mul benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # expert count sweep
        {"num_experts": 4, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 64, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},

        # max token sweep
        {"num_experts": 16, "max_tokens": 32, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 512, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},

        # hidden dimension sweep
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 1024, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 8192, "dtype": "FP32", "token_count_pattern": "random"},

        # valid token pattern sweep
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "full"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "sparse"},
    ],
    "dtype_sweep": [
        # random valid-token pattern
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP16", "token_count_pattern": "random"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "BF16", "token_count_pattern": "random"},

        # full valid-token pattern
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "full"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "FP16", "token_count_pattern": "full"},
        {"num_experts": 16, "max_tokens": 128, "hidden_dim": 4096, "dtype": "BF16", "token_count_pattern": "full"},

        # sparse valid-token pattern
        {"num_experts": 16, "max_tokens": 512, "hidden_dim": 4096, "dtype": "FP32", "token_count_pattern": "sparse"},
        {"num_experts": 16, "max_tokens": 512, "hidden_dim": 4096, "dtype": "FP16", "token_count_pattern": "sparse"},
        {"num_experts": 16, "max_tokens": 512, "hidden_dim": 4096, "dtype": "BF16", "token_count_pattern": "sparse"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_experts": [4, 8, 16, 32, 64],
    "max_tokens": [16, 32, 128, 256, 512],
    "hidden_dim": [1024, 2048, 4096, 8192, 11008],
    "dtype": ["FP32", "FP16", "BF16"],
    "token_count_pattern": ["full", "random", "sparse"],
}
