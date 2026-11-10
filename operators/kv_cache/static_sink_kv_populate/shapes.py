"""Shape templates for static_sink_kv_populate benchmark cases."""

SINGLE_SHAPES = {
    "default": {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sink length sweep
        {"sink_len": 1, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 16, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},

        # block and cache size sweep
        {"sink_len": 4, "block_size": 16, "num_kv_heads": 8, "head_dim": 128, "total_slots": 1024, "dtype": "FP32"},
        {"sink_len": 8, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 16, "block_size": 256, "num_kv_heads": 8, "head_dim": 128, "total_slots": 8192, "dtype": "FP32"},

        # head layout sweep
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 4, "head_dim": 64, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 8, "block_size": 128, "num_kv_heads": 16, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard sink populate
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "FP16"},
        {"sink_len": 4, "block_size": 128, "num_kv_heads": 8, "head_dim": 128, "total_slots": 4096, "dtype": "BF16"},

        # longer sink prefix
        {"sink_len": 16, "block_size": 256, "num_kv_heads": 8, "head_dim": 128, "total_slots": 8192, "dtype": "FP32"},
        {"sink_len": 16, "block_size": 256, "num_kv_heads": 8, "head_dim": 128, "total_slots": 8192, "dtype": "FP16"},
        {"sink_len": 16, "block_size": 256, "num_kv_heads": 8, "head_dim": 128, "total_slots": 8192, "dtype": "BF16"},

        # wider KV layout
        {"sink_len": 8, "block_size": 128, "num_kv_heads": 16, "head_dim": 128, "total_slots": 4096, "dtype": "FP32"},
        {"sink_len": 8, "block_size": 128, "num_kv_heads": 16, "head_dim": 128, "total_slots": 4096, "dtype": "FP16"},
        {"sink_len": 8, "block_size": 128, "num_kv_heads": 16, "head_dim": 128, "total_slots": 4096, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "sink_len": [1, 2, 4, 8, 16, 32],
    "block_size": [16, 32, 64, 128, 256, 512],
    "num_kv_heads": [1, 4, 8, 16],
    "head_dim": [64, 80, 96, 128, 256],
    "total_slots": [512, 1024, 2048, 4096, 8192, 32000],
    "dtype": ["FP32", "FP16", "BF16"],
}
