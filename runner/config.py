"""Runtime configuration for op_benchmark."""

import os


ARCH_LIST = ["8.0"]
NUM_PERF_TRIALS = int(os.environ.get("OP_EVAL_CUDA_NUM_TRIALS", "5"))
NUM_WARMUP = int(os.environ.get("OP_EVAL_CUDA_NUM_WARMUP", "3"))
SEED_NUM = int(os.environ.get("OP_EVAL_PERF_SEED_BASE", "1024"))
