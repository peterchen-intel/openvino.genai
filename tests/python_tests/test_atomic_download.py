# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from utils.atomic_download import AtomicDownloadManager, is_openvino_model_dir


def test_is_openvino_model_dir_empty_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert not is_openvino_model_dir(model_dir)


def test_is_openvino_model_dir_xml_only(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("xml")
    assert not is_openvino_model_dir(model_dir)


def test_is_openvino_model_dir_bin_only(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_model.bin").write_text("bin")
    assert not is_openvino_model_dir(model_dir)


def test_is_openvino_model_dir_nested(tmp_path: Path) -> None:
    model_dir = tmp_path / "model" / "nested"
    model_dir.mkdir(parents=True)
    (model_dir / "openvino_model.xml").write_text("xml")
    (model_dir / "openvino_model.bin").write_text("bin")
    assert is_openvino_model_dir(tmp_path / "model")


def test_concurrent_incomplete_destination_rejected(tmp_path: Path) -> None:
    final_path = tmp_path / "model"
    final_path.mkdir()
    (final_path / "openvino_model.xml").write_text("partial")

    manager = AtomicDownloadManager(final_path, is_valid_fn=is_openvino_model_dir)
    assert not manager.is_complete()

    with pytest.raises(FileExistsError, match="incomplete"):
        manager.execute(lambda _temp_path: None)
