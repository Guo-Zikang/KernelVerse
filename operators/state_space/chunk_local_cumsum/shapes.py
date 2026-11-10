"""Shape templates for chunk_local_cumsum benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
    "reverse": {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": True, "dtype": "FP32"},
    "vector": {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "width": 64, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 4, "seq_len": 128, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 4096, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},

        # chunk size sweep
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 32, "reverse": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 128, "reverse": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 4096, "num_heads": 16, "chunk_size": 256, "reverse": False, "dtype": "FP32"},

        # scalar, reverse, and vector forms
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": True, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "width": 64, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # forward scalar chunk cumsum
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": False, "dtype": "BF16"},

        # reverse scalar chunk cumsum
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": True, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": True, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "chunk_size": 64, "reverse": True, "dtype": "BF16"},

        # vector chunk cumsum
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "width": 64, "chunk_size": 64, "reverse": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "width": 64, "chunk_size": 64, "reverse": False, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "width": 64, "chunk_size": 64, "reverse": False, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "seq_len": [64, 128, 256, 512, 1024, 2048, 4096],
    "num_heads": [1, 4, 8, 16, 32],
    "width": [1, 16, 32, 64, 128],
    "chunk_size": [16, 32, 64, 128, 256],
    "reverse": [False, True],
    "dtype": ["FP32", "FP16", "BF16"],
}
