"""Shape templates for logits_processor_scale_softcap benchmark cases."""

SINGLE_SHAPES = {
    "default": {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
}

SHAPE_FAMILIES = {
    "shape_sweep": [
        # sequence-count sweep
        {"num_seqs": 1, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 128, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},

        # vocab-size sweep
        {"num_seqs": 16, "vocab_size": 8000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 50000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 8, "vocab_size": 128000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},

        # processor-parameter sweep
        {"num_seqs": 16, "vocab_size": 32000, "scale": 0.75, "soft_cap": None, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 15.0, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.25, "soft_cap": 50.0, "dtype": "FP32"},
    ],
    "dtype_sweep": [
        # standard logits
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP16"},
        {"num_seqs": 16, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "BF16"},

        # larger batch
        {"num_seqs": 128, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP32"},
        {"num_seqs": 128, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "FP16"},
        {"num_seqs": 128, "vocab_size": 32000, "scale": 1.0, "soft_cap": 30.0, "dtype": "BF16"},

        # scaled path without soft cap
        {"num_seqs": 16, "vocab_size": 50000, "scale": 0.75, "soft_cap": None, "dtype": "FP32"},
        {"num_seqs": 16, "vocab_size": 50000, "scale": 0.75, "soft_cap": None, "dtype": "FP16"},
        {"num_seqs": 16, "vocab_size": 50000, "scale": 0.75, "soft_cap": None, "dtype": "BF16"},
    ],
}

RANDOM_SHAPE_SPACE = {
    "num_seqs": [1, 2, 4, 8, 16, 32, 64, 128],
    "vocab_size": [8000, 16000, 32000, 50000, 65536, 100000, 128000],
    "scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "soft_cap": [None, 15.0, 30.0, 50.0],
    "dtype": ["FP32", "FP16", "BF16"],
}
