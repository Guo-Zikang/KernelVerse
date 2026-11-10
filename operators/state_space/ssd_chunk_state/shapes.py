"""Shape templates for ssd_chunk_state benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence layout sweep
        {"seq_lens": [64], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [256, 128, 64], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},

        # head and group sweep
        {"seq_lens": [128, 128], "num_heads": 4, "num_groups": 1, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 16, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},

        # dimension and chunk sweep
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 32, "dstate": 8, "chunk_size": 32, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [256, 256], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [512], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 128, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},

        # option sweep
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": False, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": False},
        {"seq_lens": [256, 128], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 128, "dtype": "FP32", "dt_softplus": False, "has_dt_bias": False},
    ],
    "dtype_sweep": [
        # standard SSD chunk state
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [128, 128], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": True},

        # wider recurrent state
        {"seq_lens": [256, 128], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 128, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [256, 128], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 128, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [256, 128], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 128, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": True},

        # no dt bias
        {"seq_lens": [256, 256], "num_heads": 16, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": False, "has_dt_bias": False},
        {"seq_lens": [256, 256], "num_heads": 16, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": False, "has_dt_bias": False},
        {"seq_lens": [256, 256], "num_heads": 16, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": False, "has_dt_bias": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_lens": [[64], [128], [256], [64, 64], [128, 128], [256, 128], [256, 256], [512], [512, 256]],
    "num_heads": [1, 4, 8, 16],
    "num_groups": [1, 2, 4, 8],
    "head_dim": [32, 64, 128],
    "dstate": [8, 16, 32],
    "chunk_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
    "dt_softplus": [True, False],
    "has_dt_bias": [True, False],
}
