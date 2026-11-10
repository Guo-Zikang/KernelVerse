"""Shape templates for apply_penalties benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_seqs": 16, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence-count sweep
        {"num_seqs": 1, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 128, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},

        # vocab-size sweep
        {"num_seqs": 16, "vocab_size": 8000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 50000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 8, "vocab_size": 128000, "seq_len": 128, "dtype": "FP32"},

        # history-length sweep
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 16, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 512, "dtype": "FP32"},
        {"num_seqs": 8, "vocab_size": 32000, "seq_len": 2048, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard vocab
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 128, "dtype": "FP16"},
        {"num_seqs": 16, "vocab_size": 32000, "seq_len": 128, "dtype": "BF16"},

        # larger batch
        {"num_seqs": 128, "vocab_size": 32000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 128, "vocab_size": 32000, "seq_len": 128, "dtype": "FP16"},
        {"num_seqs": 128, "vocab_size": 32000, "seq_len": 128, "dtype": "BF16"},

        # larger vocab
        {"num_seqs": 8, "vocab_size": 128000, "seq_len": 128, "dtype": "FP32"},
        {"num_seqs": 8, "vocab_size": 128000, "seq_len": 128, "dtype": "FP16"},
        {"num_seqs": 8, "vocab_size": 128000, "seq_len": 128, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_seqs": [1, 2, 4, 8, 16, 32, 64, 128],
    "vocab_size": [8000, 16000, 32000, 50000, 65536, 100000, 128000],
    "seq_len": [1, 8, 16, 32, 128, 512, 1024, 2048],
    "dtype": ["FP32", "FP16", "BF16"],
}
