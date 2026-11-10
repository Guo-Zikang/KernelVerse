"""Shape templates for mla_decode benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 1, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # decode batch and context sweep
        {"batch_size": 1, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 32, "seq_len": 256, "num_q_heads": 128, "block_size": 64, "dtype": "FP32"},

        # query head count sweep
        {"batch_size": 8, "seq_len": 256, "num_q_heads": 16, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 256, "num_q_heads": 32, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 256, "num_q_heads": 64, "block_size": 64, "dtype": "FP32"},

        # block size sweep
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 64, "block_size": 1, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 64, "block_size": 32, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 64, "block_size": 128, "dtype": "FP32"},

        # long context
        {"batch_size": 1, "seq_len": 2048, "num_q_heads": 32, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 4096, "num_q_heads": 32, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 16, "seq_len": 8192, "num_q_heads": 16, "block_size": 64, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard MLA decode
        {"batch_size": 1, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 64, "num_q_heads": 128, "block_size": 64, "dtype": "BF16"},

        # medium batch/context
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 32, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 32, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 512, "num_q_heads": 32, "block_size": 64, "dtype": "BF16"},

        # long context
        {"batch_size": 8, "seq_len": 4096, "num_q_heads": 32, "block_size": 64, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 4096, "num_q_heads": 32, "block_size": 64, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 4096, "num_q_heads": 32, "block_size": 64, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32],
    "seq_len": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "num_q_heads": [16, 32, 64, 128],
    "block_size": [1, 16, 32, 64, 128],
    "dtype": ["FP32", "FP16", "BF16"],
}
