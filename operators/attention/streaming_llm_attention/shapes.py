"""Shape templates for streaming_llm_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 2048,
        "num_heads": 16,
        "head_dim": 64,
        "sink_size": 4,
        "window_size": 256,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 1, "seq_len": 512, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 128, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 4096, "num_heads": 16, "head_dim": 128, "sink_size": 8, "window_size": 512, "dtype": "FP32"},

        # sink token count sweep
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 1, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 8, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 16, "window_size": 256, "dtype": "FP32"},

        # recent window sweep
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 64, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 128, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 512, "dtype": "FP32"},

        # batch and hidden size sweep
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 32, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 1024, "num_heads": 16, "head_dim": 128, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium context
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "BF16"},

        # larger batch
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "num_heads": 16, "head_dim": 64, "sink_size": 4, "window_size": 256, "dtype": "BF16"},

        # long context
        {"batch_size": 1, "seq_len": 4096, "num_heads": 16, "head_dim": 128, "sink_size": 8, "window_size": 512, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 4096, "num_heads": 16, "head_dim": 128, "sink_size": 8, "window_size": 512, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 4096, "num_heads": 16, "head_dim": 128, "sink_size": 8, "window_size": 512, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "seq_len": [512, 1024, 2048, 4096],
    "num_heads": [8, 16, 32],
    "head_dim": [64, 96, 128],
    "sink_size": [1, 2, 4, 8, 16],
    "window_size": [64, 128, 256, 512],
    "dtype": ["FP32", "FP16", "BF16"],
}
