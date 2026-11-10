"""Shape templates for selective_scan benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 2, "dim": 128, "seqlen": 256, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch size sweep
        {"batch_size": 1, "dim": 128, "seqlen": 256, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 4, "dim": 128, "seqlen": 256, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 16, "dim": 128, "seqlen": 256, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},

        # sequence length sweep
        {"batch_size": 2, "dim": 128, "seqlen": 128, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 128, "seqlen": 1024, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 128, "seqlen": 2048, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},

        # channel and state sweep
        {"batch_size": 2, "dim": 64, "seqlen": 512, "dstate": 8, "n_groups": 4, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 256, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 512, "seqlen": 512, "dstate": 16, "n_groups": 16, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},

        # option sweep
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": False, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": False, "has_D": True, "use_initial_state": False},
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": False, "use_initial_state": False},
    ],
    "dtype_sweep": [
        # standard selective scan
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP16", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},
        {"batch_size": 2, "dim": 128, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "BF16", "delta_softplus": True, "has_z": True, "has_D": True, "use_initial_state": True},

        # no optional gate
        {"batch_size": 2, "dim": 256, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": False, "has_D": True, "use_initial_state": False},
        {"batch_size": 2, "dim": 256, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "FP16", "delta_softplus": True, "has_z": False, "has_D": True, "use_initial_state": False},
        {"batch_size": 2, "dim": 256, "seqlen": 512, "dstate": 16, "n_groups": 8, "dtype": "BF16", "delta_softplus": True, "has_z": False, "has_D": True, "use_initial_state": False},

        # longer sequence
        {"batch_size": 1, "dim": 128, "seqlen": 2048, "dstate": 16, "n_groups": 8, "dtype": "FP32", "delta_softplus": True, "has_z": True, "has_D": False, "use_initial_state": True},
        {"batch_size": 1, "dim": 128, "seqlen": 2048, "dstate": 16, "n_groups": 8, "dtype": "FP16", "delta_softplus": True, "has_z": True, "has_D": False, "use_initial_state": True},
        {"batch_size": 1, "dim": 128, "seqlen": 2048, "dstate": 16, "n_groups": 8, "dtype": "BF16", "delta_softplus": True, "has_z": True, "has_D": False, "use_initial_state": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "dim": [64, 128, 256],
    "seqlen": [128, 256, 512, 1024],
    "dstate": [8, 16],
    "n_groups": [1, 2, 4, 8, 16],
    "dtype": ["FP32", "FP16", "BF16"],
    "delta_softplus": [True, False],
    "has_z": [True, False],
    "has_D": [True, False],
    "use_initial_state": [True, False],
}
