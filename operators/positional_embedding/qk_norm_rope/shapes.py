"""Shape templates for qk_norm_rope benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_tokens": 8, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"num_tokens": 1, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 8, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 256, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},

        # head grouping sweep
        {"num_tokens": 8, "num_heads_q": 16, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 64, "rotary_dim": 64, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 8, "num_heads_q": 40, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 8, "num_heads_q": 12, "num_heads_k": 1, "num_heads_v": 1, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},

        # rotary layout and partial rotary sweep
        {"num_tokens": 32, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 64, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 32, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": False},
        {"num_tokens": 32, "num_heads_q": 64, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
    ],
    "dtype_sweep": [
        # decode-style case
        {"num_tokens": 8, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 8, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP16", "is_neox": True},
        {"num_tokens": 8, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "BF16", "is_neox": True},

        # prefill-style case
        {"num_tokens": 256, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 256, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "FP16", "is_neox": True},
        {"num_tokens": 256, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 128, "eps": 1e-5, "dtype": "BF16", "is_neox": True},

        # partial rotary case
        {"num_tokens": 32, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 64, "eps": 1e-5, "dtype": "FP32", "is_neox": True},
        {"num_tokens": 32, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 64, "eps": 1e-5, "dtype": "FP16", "is_neox": True},
        {"num_tokens": 32, "num_heads_q": 32, "num_heads_k": 8, "num_heads_v": 8, "head_dim": 128, "rotary_dim": 64, "eps": 1e-5, "dtype": "BF16", "is_neox": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_tokens": [1, 8, 32, 128, 256, 512],
    "num_heads_q": [8, 12, 16, 32, 40, 64],
    "num_heads_k": [1, 4, 8],
    "num_heads_v": [1, 4, 8],
    "head_dim": [64, 128],
    "rotary_dim": [32, 64, 128],
    "eps": [1e-5, 1e-6],
    "dtype": ["FP32", "FP16", "BF16"],
    "is_neox": [True, False],
}
