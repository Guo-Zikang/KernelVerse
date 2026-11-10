"""Shape templates for cross_attention benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "batch_size": 1,
        "seq_len": 512,
        "context_len": 1024,
        "num_heads": 16,
        "head_dim": 64,
        "bias": True,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # query length sweep with fixed context
        {"batch_size": 1, "seq_len": 64, "context_len": 1024, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 128, "context_len": 1024, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 256, "context_len": 1024, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 512, "context_len": 1024, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},

        # context length sweep with fixed query
        {"batch_size": 1, "seq_len": 128, "context_len": 256, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 128, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 128, "context_len": 2048, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 128, "context_len": 4096, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},

        # hidden size and batch sweep
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 32, "head_dim": 128, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 32, "head_dim": 128, "bias": True, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 512, "context_len": 1024, "num_heads": 40, "head_dim": 128, "bias": True, "dtype": "FP32"},

        # projection bias mode sweep
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # medium encoder-decoder case
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "BF16"},

        # short query and long context
        {"batch_size": 1, "seq_len": 128, "context_len": 2048, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 128, "context_len": 2048, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 128, "context_len": 2048, "num_heads": 16, "head_dim": 64, "bias": True, "dtype": "BF16"},

        # large hidden
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 32, "head_dim": 128, "bias": True, "dtype": "FP32"},
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 32, "head_dim": 128, "bias": True, "dtype": "FP16"},
        {"batch_size": 1, "seq_len": 256, "context_len": 512, "num_heads": 32, "head_dim": 128, "bias": True, "dtype": "BF16"},

        # no projection bias
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": False, "dtype": "FP32"},
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": False, "dtype": "FP16"},
        {"batch_size": 2, "seq_len": 256, "context_len": 512, "num_heads": 16, "head_dim": 64, "bias": False, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8],
    "seq_len": [32, 64, 128, 256, 512],
    "context_len": [128, 256, 512, 1024, 2048, 4096],
    "num_heads": [8, 16, 32, 40],
    "head_dim": [64, 96, 128],
    "bias": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
