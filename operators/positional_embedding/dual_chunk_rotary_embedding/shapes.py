"""Shape templates for dual-chunk rotary embedding benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 128,
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_size": 64,
        "rotary_dim": 64,
        "dtype": "FP32",
        "chunk_size": 64,
        "local_size": 16,
        "is_neox_style": True,
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode token-count sweep
        {"batch_size": 32, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 512, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},

        # prefill sequence length sweep
        {"batch_size": 1, "seq_len": 128, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 128, "local_size": 32, "is_neox_style": True},
        {"batch_size": 4, "seq_len": 1024, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 256, "local_size": 64, "is_neox_style": True},

        # chunk/local-size sweep
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 32, "local_size": 8, "is_neox_style": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 128, "local_size": 32, "is_neox_style": True},

        # layout and head-size sweep
        {"batch_size": 2, "seq_len": 256, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": False},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 128, "local_size": 32, "is_neox_style": False},
        {"batch_size": 1, "seq_len": 1024, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP32", "chunk_size": 256, "local_size": 64, "is_neox_style": False},
    ],
    "dtype_sweep": [
        # decode case
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP16", "chunk_size": 64, "local_size": 16, "is_neox_style": True},
        {"batch_size": 128, "seq_len": 1, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "BF16", "chunk_size": 64, "local_size": 16, "is_neox_style": True},

        # prefill case
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP32", "chunk_size": 128, "local_size": 32, "is_neox_style": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "FP16", "chunk_size": 128, "local_size": 32, "is_neox_style": True},
        {"batch_size": 2, "seq_len": 512, "num_q_heads": 32, "num_kv_heads": 8, "head_size": 64, "rotary_dim": 64, "dtype": "BF16", "chunk_size": 128, "local_size": 32, "is_neox_style": True},

        # wider head case
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP32", "chunk_size": 128, "local_size": 32, "is_neox_style": False},
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "FP16", "chunk_size": 128, "local_size": 32, "is_neox_style": False},
        {"batch_size": 1, "seq_len": 512, "num_q_heads": 64, "num_kv_heads": 8, "head_size": 128, "rotary_dim": 128, "dtype": "BF16", "chunk_size": 128, "local_size": 32, "is_neox_style": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 32, 128],
    "seq_len": [1, 64, 128, 256, 512, 1024],
    "num_q_heads": [16, 32, 64],
    "num_kv_heads": [8, 16],
    "head_size": [64, 128],
    "rotary_dim": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
    "base": [10000.0],
    "chunk_size": [32, 64, 128, 256],
    "local_size": [8, 16, 32, 64],
    "is_neox_style": [True, False],
}
