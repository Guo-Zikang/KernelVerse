"""Shape templates for reorg_kvcache benchmark cases."""

SINGLE_SHAPES = {
    "default": {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # world size sweep
        {"world_size": 1, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 8, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},

        # sequence count sweep
        {"world_size": 2, "num_sequences": 1, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 32, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 128, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},

        # feature dimension sweep
        {"world_size": 2, "num_sequences": 16, "feature_dim": 128, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 512, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},

        # chunk position sweep
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 64, "chunk_size": 64, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 1, "dtype": "FP32"},
        {"world_size": 4, "num_sequences": 32, "feature_dim": 512, "padded_chunk_len": 256, "chunk_size": 128, "chunk_idx": 2, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard reorganization
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP16"},
        {"world_size": 2, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "BF16"},

        # larger world size
        {"world_size": 8, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP32"},
        {"world_size": 8, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "FP16"},
        {"world_size": 8, "num_sequences": 16, "feature_dim": 576, "padded_chunk_len": 128, "chunk_size": 128, "chunk_idx": 0, "dtype": "BF16"},

        # larger chunk
        {"world_size": 4, "num_sequences": 32, "feature_dim": 512, "padded_chunk_len": 256, "chunk_size": 128, "chunk_idx": 1, "dtype": "FP32"},
        {"world_size": 4, "num_sequences": 32, "feature_dim": 512, "padded_chunk_len": 256, "chunk_size": 128, "chunk_idx": 1, "dtype": "FP16"},
        {"world_size": 4, "num_sequences": 32, "feature_dim": 512, "padded_chunk_len": 256, "chunk_size": 128, "chunk_idx": 1, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "world_size": [1, 2, 4, 8],
    "num_sequences": [1, 8, 16, 32, 128],
    "feature_dim": [128, 512, 576, 1024],
    "padded_chunk_len": [16, 64, 128, 256],
    "chunk_size": [64, 128, 256],
    "chunk_idx": [0, 1, 2],
    "dtype": ["FP32", "FP16", "BF16"],
}
