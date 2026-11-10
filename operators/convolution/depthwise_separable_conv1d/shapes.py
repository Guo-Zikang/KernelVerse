"""Shape templates for depthwise_separable_conv1d benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence length sweep
        {"batch_size": 4, "seq_len": 512, "input_dim": 128, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 256, "padding": 1, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "FP32"},
        {"batch_size": 16, "seq_len": 4096, "input_dim": 512, "depthwise_multiplier": 1, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "FP32"},

        # kernel and multiplier sweep
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 1, "kernel_size": 3, "out_channels": 256, "padding": 1, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 5, "out_channels": 512, "padding": 2, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "depthwise_multiplier": 4, "kernel_size": 7, "out_channels": 1024, "padding": 3, "dtype": "FP32"},

        # channel projection sweep
        {"batch_size": 16, "seq_len": 1024, "input_dim": 128, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 128, "padding": 1, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 1024, "padding": 1, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 4096, "input_dim": 1024, "depthwise_multiplier": 1, "kernel_size": 3, "out_channels": 4096, "padding": 1, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard depthwise separable convolution
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 512, "padding": 1, "dtype": "BF16"},

        # wider kernel
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 5, "out_channels": 512, "padding": 2, "dtype": "FP32"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 5, "out_channels": 512, "padding": 2, "dtype": "FP16"},
        {"batch_size": 8, "seq_len": 2048, "input_dim": 256, "depthwise_multiplier": 2, "kernel_size": 5, "out_channels": 512, "padding": 2, "dtype": "BF16"},

        # pointwise expansion
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 2048, "padding": 1, "dtype": "FP32"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 2048, "padding": 1, "dtype": "FP16"},
        {"batch_size": 4, "seq_len": 1024, "input_dim": 512, "depthwise_multiplier": 2, "kernel_size": 3, "out_channels": 2048, "padding": 1, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32],
    "seq_len": [128, 256, 512, 1024, 2048, 4096, 8192],
    "input_dim": [64, 128, 256, 512, 1024],
    "depthwise_multiplier": [1, 2, 4],
    "kernel_size": [3, 5, 7],
    "out_channels": [64, 128, 256, 512, 1024, 2048, 4096],
    "padding": [0, 1, 2, 3],
    "dtype": ["FP32", "FP16", "BF16"],
}
