"""Shared Biren SDK environment setup."""

from __future__ import annotations

import os
import sys
from typing import Optional


BIREN_SDK_ROOTS = (
    "/usr/local/birensupa/sdk/latest",
    "/usr/local/birensupa/sdk/1.10.0.0.beta1",
)


def _prepend_environment_path(variable_name: str, path: str) -> None:
    entries = [entry for entry in os.environ.get(variable_name, "").split(os.pathsep) if entry]
    if path not in entries:
        entries.insert(0, path)
        os.environ[variable_name] = os.pathsep.join(entries)


def prepare_biren_environment(device_id: Optional[int] = None) -> Optional[str]:
    """Configure the current process and future subprocesses for the Biren SDK."""
    if device_id is not None:
        os.environ["SUPA_VISIBLE_DEVICES"] = str(device_id)

    sdk_root = next((root for root in BIREN_SDK_ROOTS if os.path.isdir(root)), None)
    if sdk_root is None:
        return None

    sulib_dir = os.path.join(sdk_root, "sulib", "lib")
    if os.path.isdir(sulib_dir):
        if sulib_dir not in sys.path:
            sys.path.insert(0, sulib_dir)
        _prepend_environment_path("PYTHONPATH", sulib_dir)

    brcc_bin_dir = os.path.join(sdk_root, "brcc", "bin")
    if os.path.isdir(brcc_bin_dir):
        _prepend_environment_path("PATH", brcc_bin_dir)

    return sdk_root
