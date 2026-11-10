"""Shape templates for glu_pointwise_conv benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 4, "seq_len": 512, "input_dim": 128, "output_dim": 256, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
        {"batch_size": 16, "seq_len": 4096, "input_dim": 512, "output_dim": 1024, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},

        # causal and kernel sweep
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 1, "bias_in_glu": False, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": True, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "output_dim": 1024, "kernel_size": 5, "bias_in_glu": True, "causal": True, "dtype": "FP32"},

        # channel projection sweep
        {"batch_size": 16, "seq_len": 1024, "input_dim": 128, "output_dim": 128, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 1024, "kernel_size": 3, "bias_in_glu": False, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 4096, "input_dim": 1024, "output_dim": 4096, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # non-causal GLU convolution
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": False, "dtype": "BF16"},

        # causal GLU convolution
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": True, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": True, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "output_dim": 512, "kernel_size": 3, "bias_in_glu": True, "causal": True, "dtype": "BF16"},

        # no extra GLU bias
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "output_dim": 2048, "kernel_size": 5, "bias_in_glu": False, "causal": False, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "output_dim": 2048, "kernel_size": 5, "bias_in_glu": False, "causal": False, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "output_dim": 2048, "kernel_size": 5, "bias_in_glu": False, "causal": False, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32],
    "seq_len": [128, 256, 512, 1024, 2048, 4096, 8192],
    "input_dim": [64, 128, 256, 512, 1024],
    "output_dim": [64, 128, 256, 512, 1024, 2048, 4096],
    "kernel_size": [1, 3, 5, 7],
    "bias_in_glu": [True, False],
    "causal": [True, False],
    "dtype": ["FP32", "FP16", "BF16"],
}
