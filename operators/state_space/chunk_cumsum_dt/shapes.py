"""Shape templates for chunk_cumsum_dt benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence layout sweep
        {"seq_lens": [128, 128, 128, 128], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [4096, 2048, 1024, 512], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},

        # head-count sweep
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 8, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 32, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [2048, 2048, 2048, 2048], "num_heads": 64, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},

        # chunk-size sweep
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 32, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 128, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [4096, 4096], "num_heads": 32, "chunk_size": 256, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
    ],
    "dtype_sweep": [
        # standard dt preprocessing
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": True},

        # no dt bias
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": False},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": False},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": False},

        # no softplus
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": False, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": False, "has_dt_bias": True},
        {"seq_lens": [1024, 1024, 1024, 1024], "num_heads": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": False, "has_dt_bias": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_lens": [[128], [512], [1024], [128, 128, 128, 128], [512, 512, 512, 512], [1024, 1024, 1024, 1024], [4096, 2048, 1024, 512]],
    "num_heads": [1, 4, 8, 16, 32, 64],
    "chunk_size": [16, 32, 64, 128, 256],
    "dtype": ["FP32", "FP16", "BF16"],
    "dt_softplus": [True, False],
    "has_dt_bias": [True, False],
}
