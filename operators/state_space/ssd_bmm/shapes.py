"""Shape templates for ssd_bmm benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence layout sweep
        {"seq_lens": [64], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [256, 128, 64], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},

        # chunk size sweep
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 32, "dtype": "FP32", "causal": True},
        {"seq_lens": [256, 256], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [512, 256], "num_groups": 8, "key_dim": 64, "chunk_size": 128, "dtype": "FP32", "causal": True},

        # group and key dimension sweep
        {"seq_lens": [128, 128], "num_groups": 4, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 16, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 128, "chunk_size": 64, "dtype": "FP32", "causal": True},

        # causal behavior sweep
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": False},
        {"seq_lens": [256, 256], "num_groups": 8, "key_dim": 128, "chunk_size": 128, "dtype": "FP32", "causal": False},
        {"seq_lens": [512], "num_groups": 16, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": False},
    ],
    "dtype_sweep": [
        # standard causal SSD BMM
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP16", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "BF16", "causal": True},

        # non-causal block Gram
        {"seq_lens": [256, 256], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP32", "causal": False},
        {"seq_lens": [256, 256], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "FP16", "causal": False},
        {"seq_lens": [256, 256], "num_groups": 8, "key_dim": 64, "chunk_size": 64, "dtype": "BF16", "causal": False},

        # wider key dimension
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 128, "chunk_size": 64, "dtype": "FP32", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 128, "chunk_size": 64, "dtype": "FP16", "causal": True},
        {"seq_lens": [128, 128], "num_groups": 8, "key_dim": 128, "chunk_size": 64, "dtype": "BF16", "causal": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_lens": [[64], [128], [256], [64, 64], [128, 128], [256, 128], [256, 256], [512], [512, 256]],
    "num_groups": [1, 4, 8, 16],
    "key_dim": [32, 64, 128],
    "chunk_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
    "causal": [True, False],
}
