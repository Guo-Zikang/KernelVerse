"""Shape templates for per-tensor affine quantization benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "shape": [128, 4096],
        "scale": 0.05,
        "zero_point": 0,
        "quantized_dtype": "QINT8",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # hidden size sweep
        {"shape": [1, 1024], "scale": 0.02, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [128, 1024], "scale": 0.04, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [128, 4096], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [512, 4096], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [512, 8192], "scale": 0.08, "zero_point": 0, "quantized_dtype": "QINT8"},

        # batched sequence
        {"shape": [2, 128, 1024], "scale": 0.03, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [4, 512, 4096], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [8, 1024, 4096], "scale": 0.08, "zero_point": 0, "quantized_dtype": "QINT8"},

        # decode-style tensors
        {"shape": [16, 1, 4096], "scale": 0.04, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [128, 1, 8192], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [256, 1, 4096], "scale": 0.08, "zero_point": 0, "quantized_dtype": "QINT8"},
    ],
    "dtype_sweep": [
        # hidden vector
        {"shape": [128, 4096], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [128, 4096], "scale": 0.05, "zero_point": 128, "quantized_dtype": "QUINT8"},
        {"shape": [128, 4096], "scale": 0.08, "zero_point": 137, "quantized_dtype": "QUINT8"},

        # batched sequence
        {"shape": [4, 512, 4096], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
        {"shape": [4, 512, 4096], "scale": 0.05, "zero_point": 128, "quantized_dtype": "QUINT8"},
        {"shape": [4, 512, 4096], "scale": 0.02, "zero_point": -3, "quantized_dtype": "QINT8"},

        # scale and zero-point variation
        {"shape": [128, 1024], "scale": 0.02, "zero_point": -3, "quantized_dtype": "QINT8"},
        {"shape": [128, 1024], "scale": 0.08, "zero_point": 137, "quantized_dtype": "QUINT8"},
        {"shape": [128, 1024], "scale": 0.05, "zero_point": 0, "quantized_dtype": "QINT8"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "shape": [
        [1, 1024],
        [1, 4096],
        [1, 8192],
        [128, 1024],
        [128, 4096],
        [512, 4096],
        [2048, 4096],
        [512, 8192],
        [2, 128, 1024],
        [4, 512, 4096],
        [8, 1024, 4096],
        [16, 64, 32],
        [8, 128, 64],
        [4, 256, 128],
        [2, 512, 256],
        [32, 16, 64],
        [2, 16, 64, 32],
        [4, 32, 32, 64],
        [16, 1, 4096],
        [128, 1, 8192],
    ],
    "scale": [0.02, 0.04, 0.05, 0.08],
    "zero_point": [0],
    "quantized_dtype": ["QINT8"],
}
