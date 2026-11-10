"""Shape templates for embedding lookup benchmark cases."""

SINGLE_SHAPES = {
    "default": {
        "input_shape": [128, 128],
        "num_embeddings": 32000,
        "embedding_dim": 4096,
        "dtype": "FP32",
    },
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # token count sweep
        {"input_shape": [1, 1], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [1, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [128, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [16, 2048], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},

        # vocabulary size sweep
        {"input_shape": [8, 128], "num_embeddings": 8192, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [8, 128], "num_embeddings": 128000, "embedding_dim": 4096, "dtype": "FP32"},

        # embedding dimension sweep
        {"input_shape": [32, 196], "num_embeddings": 8192, "embedding_dim": 768, "dtype": "FP32"},
        {"input_shape": [64, 256], "num_embeddings": 16384, "embedding_dim": 1024, "dtype": "FP32"},
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [16, 2048], "num_embeddings": 128000, "embedding_dim": 8192, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # LLM token embedding
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP32"},
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "FP16"},
        {"input_shape": [8, 128], "num_embeddings": 32000, "embedding_dim": 4096, "dtype": "BF16"},

        # large vocabulary
        {"input_shape": [16, 2048], "num_embeddings": 128000, "embedding_dim": 8192, "dtype": "FP32"},
        {"input_shape": [16, 2048], "num_embeddings": 128000, "embedding_dim": 8192, "dtype": "FP16"},
        {"input_shape": [16, 2048], "num_embeddings": 128000, "embedding_dim": 8192, "dtype": "BF16"},

        # vision-like tokens
        {"input_shape": [64, 256], "num_embeddings": 16384, "embedding_dim": 1024, "dtype": "FP32"},
        {"input_shape": [64, 256], "num_embeddings": 16384, "embedding_dim": 1024, "dtype": "FP16"},
        {"input_shape": [64, 256], "num_embeddings": 16384, "embedding_dim": 1024, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "input_shape": [
        [1, 1],
        [1, 128],
        [8, 128],
        [16, 2048],
        [32, 196],
        [64, 256],
        [128, 128],
        [128, 576],
    ],
    "num_embeddings": [8192, 16384, 32000, 32768, 128000],
    "embedding_dim": [768, 1024, 4096, 8192],
    "dtype": ["FP32", "FP16", "BF16"],
}
