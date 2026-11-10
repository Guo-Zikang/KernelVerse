"""Shape templates for mamba_chunk_scan_combined_varlen benchmark cases."""

SINGLE_SHAPES = {
    "default": {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # variable-length sequence sweep
        {"seq_lens": [32, 48], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 32, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [128, 160, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},

        # head, group, and state sweep
        {"seq_lens": [64, 64], "num_heads": 4, "num_groups": 1, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 64], "num_heads": 8, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 64], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},

        # optional path sweep
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": False, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": False, "has_D": False, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": False, "has_initial_states": False},
    ],
    "dtype_sweep": [
        # standard varlen scan
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 96], "num_heads": 8, "num_groups": 2, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},

        # wider state
        {"seq_lens": [64, 64], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 64], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},
        {"seq_lens": [64, 64], "num_heads": 8, "num_groups": 2, "head_dim": 128, "dstate": 32, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": True, "has_D": True, "has_z": True, "has_initial_states": True},

        # optional features disabled
        {"seq_lens": [128, 96], "num_heads": 8, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP32", "dt_softplus": True, "has_dt_bias": False, "has_D": False, "has_z": False, "has_initial_states": False},
        {"seq_lens": [128, 96], "num_heads": 8, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "FP16", "dt_softplus": True, "has_dt_bias": False, "has_D": False, "has_z": False, "has_initial_states": False},
        {"seq_lens": [128, 96], "num_heads": 8, "num_groups": 4, "head_dim": 64, "dstate": 16, "chunk_size": 64, "dtype": "BF16", "dt_softplus": True, "has_dt_bias": False, "has_D": False, "has_z": False, "has_initial_states": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "seq_lens": [[32], [64], [32, 48], [64, 96], [128, 96], [64, 64, 64], [128, 160, 96]],
    "num_heads": [4, 8, 16],
    "num_groups": [1, 2, 4, 8],
    "head_dim": [32, 64, 128],
    "dstate": [8, 16, 32],
    "chunk_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
    "dt_softplus": [True, False],
    "has_dt_bias": [True, False],
    "has_D": [True, False],
    "has_z": [True, False],
    "has_initial_states": [True, False],
}
