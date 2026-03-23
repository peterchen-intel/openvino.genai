# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
import sys
from pathlib import Path


def test_import_patch_pyav_for_servercore_executes_python_tests_helper():
    repository_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import patch_pyav_for_servercore as patch_module; "
            "assert patch_module.sys is sys; "
            "print(patch_module.__file__)"
        ),
    ]
    result = subprocess.run(command, cwd=repository_root, check=True, capture_output=True, text=True)

    assert Path(result.stdout.strip()).resolve() == repository_root / "tests" / "python_tests" / "patch_pyav_for_servercore.py"
