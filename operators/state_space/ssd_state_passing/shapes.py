"""Shape templates for ssd_state_passing benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # batch and chunk layout sweep
        {"batch_size": 1, "num_chunks": 8, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 8, "num_chunks": 32, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},

        # head and state dimension sweep
        {"batch_size": 4, "num_chunks": 16, "num_heads": 4, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 16, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 2048, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},

        # chunk size and initial state sweep
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 32, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 128, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": False},
    ],
    "dtype_sweep": [
        # standard state passing
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP16", "has_initial_states": True},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "BF16", "has_initial_states": True},

        # no initial states
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP32", "has_initial_states": False},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "FP16", "has_initial_states": False},
        {"batch_size": 4, "num_chunks": 16, "num_heads": 8, "state_dim": 1024, "chunk_size": 64, "dtype": "BF16", "has_initial_states": False},

        # larger state dimension
        {"batch_size": 8, "num_chunks": 32, "num_heads": 16, "state_dim": 2048, "chunk_size": 128, "dtype": "FP32", "has_initial_states": True},
        {"batch_size": 8, "num_chunks": 32, "num_heads": 16, "state_dim": 2048, "chunk_size": 128, "dtype": "FP16", "has_initial_states": True},
        {"batch_size": 8, "num_chunks": 32, "num_heads": 16, "state_dim": 2048, "chunk_size": 128, "dtype": "BF16", "has_initial_states": True},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "num_chunks": [1, 2, 4, 8, 16, 32, 64],
    "num_heads": [1, 4, 8, 16, 32],
    "state_dim": [256, 512, 1024, 2048, 4096],
    "chunk_size": [32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
    "has_initial_states": [True, False],
}
