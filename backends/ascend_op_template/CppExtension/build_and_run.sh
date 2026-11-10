#!/bin/bash
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

BASE_DIR=$(pwd)
PYBIND_DEST=${PYBIND_DEST:-${BASE_DIR}/pybind_install}
PYTHON_BIN=${PYTHON_BIN:-python3}

rm -rf "${PYBIND_DEST}"
mkdir -p "${PYBIND_DEST}"
rm -rf "${BASE_DIR}/build" "${BASE_DIR}/dist"

# 编译wheel包
if ! "${PYTHON_BIN}" setup.py build bdist_wheel; then
    echo "Error: Failed to build wheel package"
    exit 1
fi

# 安装wheel包到工作空间
cd "${BASE_DIR}/dist"
if ! "${PYTHON_BIN}" -m pip install *.whl --force-reinstall --no-deps --target "${PYBIND_DEST}"; then
    echo "Error: Failed to install wheel package into ${PYBIND_DEST}"
    exit 1
fi
