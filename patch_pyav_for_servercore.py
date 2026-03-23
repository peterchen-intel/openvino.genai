# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

module_path = Path(__file__).resolve().parent / "tests" / "python_tests" / "patch_pyav_for_servercore.py"
module_spec = spec_from_file_location(__name__, module_path)
if module_spec is None or module_spec.loader is None:
    raise ImportError(f"Cannot load {__name__} from {module_path}")

module = module_from_spec(module_spec)
sys.modules[__name__] = module
module_spec.loader.exec_module(module)
