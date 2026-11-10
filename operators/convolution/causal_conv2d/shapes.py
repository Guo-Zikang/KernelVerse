"""Shape templates for causal_conv2d benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # common width-causal convolution cases
        {"batch_size": 1, "in_channels": 32, "out_channels": 64, "height": 112, "width": 224, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
        {"batch_size": 16, "in_channels": 128, "out_channels": 256, "height": 32, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "FP32"},

        # strided width-causal convolution cases
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 256, "kernel_size": 3, "stride": 2, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 128, "out_channels": 256, "height": 64, "width": 512, "kernel_size": 5, "stride": 2, "dtype": "FP32"},
        {"batch_size": 2, "in_channels": 256, "out_channels": 512, "height": 32, "width": 512, "kernel_size": 3, "stride": 4, "dtype": "FP32"},

        # channel and kernel-size sweep cases
        {"batch_size": 16, "in_channels": 32, "out_channels": 64, "height": 128, "width": 128, "kernel_size": 1, "stride": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 128, "out_channels": 128, "height": 56, "width": 224, "kernel_size": 5, "stride": 1, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 256, "out_channels": 256, "height": 32, "width": 1024, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # regular width-causal convolution
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "FP16"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 128, "kernel_size": 3, "stride": 1, "dtype": "BF16"},

        # strided width-causal convolution
        {"batch_size": 4, "in_channels": 128, "out_channels": 256, "height": 64, "width": 512, "kernel_size": 5, "stride": 2, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 128, "out_channels": 256, "height": 64, "width": 512, "kernel_size": 5, "stride": 2, "dtype": "FP16"},
        {"batch_size": 4, "in_channels": 128, "out_channels": 256, "height": 64, "width": 512, "kernel_size": 5, "stride": 2, "dtype": "BF16"},

        # long-width feature maps
        {"batch_size": 4, "in_channels": 256, "out_channels": 256, "height": 32, "width": 1024, "kernel_size": 3, "stride": 1, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 256, "out_channels": 256, "height": 32, "width": 1024, "kernel_size": 3, "stride": 1, "dtype": "FP16"},
        {"batch_size": 4, "in_channels": 256, "out_channels": 256, "height": 32, "width": 1024, "kernel_size": 3, "stride": 1, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16],
    "in_channels": [16, 32, 64, 128, 256],
    "out_channels": [16, 32, 64, 128, 256, 512],
    "height": [16, 32, 56, 64, 112, 128],
    "width": [32, 64, 128, 224, 256, 512, 1024],
    "kernel_size": [1, 3, 5],
    "stride": [1, 2, 4],
    "dtype": ["FP32", "FP16", "BF16"],
}
