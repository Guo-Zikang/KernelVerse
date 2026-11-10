"""Shape templates for nonzero benchmark cases."""

SINGLE_SHAPES = {
    "default": {"input_dims": [1024, 4096], "dtype": "INT64", "index_dtype": "INT64", "target_nonzero_count": 4096},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # 1D mask density sweep
        {"input_dims": [1024], "dtype": "BOOL", "index_dtype": "INT64", "target_nonzero_count": 32},
        {"input_dims": [16384], "dtype": "BOOL", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [65536], "dtype": "BOOL", "index_dtype": "INT64", "target_nonzero_count": 49152},

        # 2D sparse and dense layouts
        {"input_dims": [1024, 4096], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 128},
        {"input_dims": [1024, 4096], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 1048576},

        # 3D tensor layouts
        {"input_dims": [16, 512, 1024], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [32, 1024, 1024], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 32768},
        {"input_dims": [64, 1024, 1024], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 262144},

        # 4D tensor layouts
        {"input_dims": [4, 32, 128, 128], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [8, 32, 256, 128], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 65536},
        {"input_dims": [16, 16, 512, 256], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 262144},
    ],
    "dtype_sweep": [
        # input dtype
        {"input_dims": [1024, 4096], "dtype": "BOOL", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "INT32", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "INT64", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "FP16", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "BF16", "index_dtype": "INT64", "target_nonzero_count": 4096},

        # output index dtype
        {"input_dims": [1024, 4096], "dtype": "BOOL", "index_dtype": "INT32", "target_nonzero_count": 4096},
        {"input_dims": [1024, 4096], "dtype": "BOOL", "index_dtype": "INT64", "target_nonzero_count": 4096},
        {"input_dims": [8192, 4096], "dtype": "FP32", "index_dtype": "INT32", "target_nonzero_count": 32768},
        {"input_dims": [8192, 4096], "dtype": "FP32", "index_dtype": "INT64", "target_nonzero_count": 32768},
    ],
}

RANDOM_SHAPE_SPACE = {
    "input_dims": [
        [128],
        [1024],
        [16384],
        [65536],
        [128, 2048],
        [1024, 4096],
        [8192, 4096],
        [16, 512, 1024],
        [32, 1024, 1024],
        [64, 1024, 1024],
        [4, 32, 128, 128],
        [8, 32, 256, 128],
        [16, 16, 512, 256],
    ],
    "dtype": ["BOOL", "INT32", "INT64", "FP32", "FP16", "BF16"],
    "index_dtype": ["INT32", "INT64"],
    "target_nonzero_count": [0, 1, 32, 128, 256, 512, 4096, 32768, 262144, 1048576],
}
