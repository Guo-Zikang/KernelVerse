"""Shape templates for Phi-3 LongRoPE scaled rotary embedding benchmark cases."""

SHORT_FACTOR_32 = [1.0] * 32
LONG_FACTOR_32 = [8.0] * 32
SHORT_FACTOR_64 = [1.0] * 64
LONG_FACTOR_64 = [8.0] * 64
MIXED_LONG_FACTOR_32 = [4.0] * 16 + [8.0] * 16
MIXED_LONG_FACTOR_64 = [4.0] * 32 + [8.0] * 32

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 128,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_size": 64,
        "rotary_dim": 64,
        "dtype": "FP32",
        "original_max_position_embeddings": 4096,
        "short_factor": SHORT_FACTOR_32,
        "long_factor": LONG_FACTOR_32,
        "use_long_rope": True,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode token-count sweep
        {"batch_size": 32, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 512, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},

        # prefill sequence length sweep
        {"batch_size": 1, "seq_len": 128, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 4, "seq_len": 1024, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},

        # Phi-3 LongRoPE factor and rotary-width sweep
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": False},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": MIXED_LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 1, "seq_len": 1024, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_64, "long_factor": MIXED_LONG_FACTOR_64, "use_long_rope": True},

        # mscale sweep
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True, "long_mscale": 1.0},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True, "long_mscale": 1.1},
        {"batch_size": 1, "seq_len": 1024, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_64, "long_factor": LONG_FACTOR_64, "use_long_rope": True, "long_mscale": 1.2},
    ],
    "dtype_sweep": [
        # decode case
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "BF16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},

        # prefill case
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "BF16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_32, "long_factor": LONG_FACTOR_32, "use_long_rope": True},

        # wider head case
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP32", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_64, "long_factor": LONG_FACTOR_64, "use_long_rope": True},
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_64, "long_factor": LONG_FACTOR_64, "use_long_rope": True},
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "BF16", "original_max_position_embeddings": 4096, "short_factor": SHORT_FACTOR_64, "long_factor": LONG_FACTOR_64, "use_long_rope": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 32, 128],
    "seq_len": [1, 64, 128, 256, 512, 1024],
    "num_q_heads": [16, 32, 64],
    "num_kv_heads": [8, 16],
    "head_size": [64, 128],
    "rotary_dim": [64],
    "dtype": ["FP32", "FP16", "BF16"],
    "base": [10000.0],
    "original_max_position_embeddings": [4096],
    "short_factor": [SHORT_FACTOR_32],
    "long_factor": [LONG_FACTOR_32, MIXED_LONG_FACTOR_32],
    "use_long_rope": [True, False],
    "short_mscale": [1.0],
    "long_mscale": [1.0, 1.1],
}
