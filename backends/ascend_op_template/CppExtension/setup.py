import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

MODULE_NAME = os.getenv("CUSTOM_OPS_MODULE_NAME", "custom_ops_lib")
PACKAGE_NAME = os.getenv("CUSTOM_OPS_PACKAGE_NAME", "custom_ops")

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

extension = NpuExtension(
    name=MODULE_NAME,
    sources=source_files,
    extra_compile_args=[
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)

setup(
    name=PACKAGE_NAME,
    version='1.0',
    keywords='custom_ops',
    ext_modules=[extension],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
