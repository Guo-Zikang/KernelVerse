"""Shape templates for conv2d benchmark cases."""

SINGLE_SHAPES = {
    "default": {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # common 3x3 convolution over image-like feature maps
        {"batch_size": 1, "in_channels": 32, "out_channels": 64, "height": 224, "width": 224, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 16, "in_channels": 128, "out_channels": 256, "height": 32, "width": 32, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},

        # pointwise and strided convolution cases
        {"batch_size": 16, "in_channels": 256, "out_channels": 512, "height": 32, "width": 32, "kernel_size": 1, "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 128, "out_channels": 256, "height": 128, "width": 128, "kernel_size": 3, "stride": 2, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 64, "out_channels": 128, "height": 128, "width": 128, "kernel_size": 5, "stride": 1, "padding": 2, "dilation": 1, "groups": 1, "dtype": "FP32"},

        # grouped and depthwise-style convolution cases
        {"batch_size": 16, "in_channels": 128, "out_channels": 128, "height": 56, "width": 56, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 128, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 56, "width": 56, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2, "groups": 4, "dtype": "FP32"},
        {"batch_size": 4, "in_channels": 256, "out_channels": 512, "height": 112, "width": 112, "kernel_size": 3, "stride": 2, "padding": 1, "dilation": 1, "groups": 32, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # regular 3x3 convolution
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "FP16"},
        {"batch_size": 8, "in_channels": 64, "out_channels": 128, "height": 64, "width": 64, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "dtype": "BF16"},

        # pointwise convolution
        {"batch_size": 16, "in_channels": 256, "out_channels": 512, "height": 32, "width": 32, "kernel_size": 1, "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "dtype": "FP32"},
        {"batch_size": 16, "in_channels": 256, "out_channels": 512, "height": 32, "width": 32, "kernel_size": 1, "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "dtype": "FP16"},
        {"batch_size": 16, "in_channels": 256, "out_channels": 512, "height": 32, "width": 32, "kernel_size": 1, "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "dtype": "BF16"},

        # grouped convolution
        {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 56, "width": 56, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 4, "dtype": "FP32"},
        {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 56, "width": 56, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 4, "dtype": "FP16"},
        {"batch_size": 8, "in_channels": 256, "out_channels": 256, "height": 56, "width": 56, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 4, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "batch_size": [1, 2, 4, 8, 16, 32],
    "in_channels": [16, 32, 64, 128, 256, 512],
    "out_channels": [16, 32, 64, 128, 256, 512],
    "height": [16, 32, 56, 64, 112, 128, 224],
    "width": [16, 32, 56, 64, 112, 128, 224],
    "kernel_size": [1, 3, 5],
    "stride": [1, 2],
    "padding": [0, 1, 2],
    "dilation": [1, 2],
    "groups": [1, 2, 4, 8, 16, 32, 64],
    "dtype": ["FP32", "FP16", "BF16"],
}
