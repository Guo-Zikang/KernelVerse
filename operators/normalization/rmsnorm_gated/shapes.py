"""Shape templates for rmsnorm_gated benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # hidden size sweep
        {"batch_size": 1, "hidden_dim": 1024, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
        {"batch_size": 512, "hidden_dim": 8192, "eps": 1e-6, "dtype": "FP32", "prenorm": False},

        # prenorm output-structure sweep
        {"batch_size": 1, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": True},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": True},
        {"batch_size": 512, "hidden_dim": 8192, "eps": 1e-6, "dtype": "FP32", "prenorm": True},

        # decode and wide hidden cases
        {"batch_size": 16, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
        {"batch_size": 64, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": True},
        {"batch_size": 128, "hidden_dim": 12288, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
    ],
    "dtype_sweep": [
        # non-prenorm output
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP16", "prenorm": False},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "BF16", "prenorm": False},

        # prenorm tuple output
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP32", "prenorm": True},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "FP16", "prenorm": True},
        {"batch_size": 128, "hidden_dim": 4096, "eps": 1e-6, "dtype": "BF16", "prenorm": True},

        # wide hidden tensor
        {"batch_size": 64, "hidden_dim": 8192, "eps": 1e-6, "dtype": "FP32", "prenorm": False},
        {"batch_size": 64, "hidden_dim": 8192, "eps": 1e-6, "dtype": "FP16", "prenorm": False},
        {"batch_size": 64, "hidden_dim": 8192, "eps": 1e-6, "dtype": "BF16", "prenorm": False},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 16, 64, 128, 512, 2048],
    "hidden_dim": [512, 1024, 4096, 8192, 12288],
    "eps": [1e-6, 1e-5],
    "dtype": ["FP32", "FP16", "BF16"],
    "prenorm": [True, False],
}
