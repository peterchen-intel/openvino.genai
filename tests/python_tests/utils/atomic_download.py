# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import uuid
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _path_exists(path: Path) -> bool:
    return path.exists()


class AtomicDownloadManager:
    def __init__(self, final_path: Path, is_valid_fn: Callable[[Path], bool] | None = None):
        """Create an atomic download manager.

        Args:
            final_path: Destination path for the final downloaded directory.
            is_valid_fn: Optional destination validator used by `is_complete()`.
                Crucial in concurrent CI scenarios where another process may create
                an incomplete destination before the current process promotes temp data.
        """
        self.final_path = Path(final_path)
        self.is_valid_fn = is_valid_fn or _path_exists
        self._uses_custom_validator = is_valid_fn is not None
        random_suffix = uuid.uuid4().hex[:8]
        self.temp_path = self.final_path.parent / f".tmp_{self.final_path.name}_{random_suffix}"

    def is_complete(self) -> bool:
        return self.is_valid_fn(self.final_path)

    def execute(self, download_fn: Callable[[Path], None]) -> None:
        if self.is_complete():
            logger.info(f"Already downloaded: {self.final_path}")
            return

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        try:
            download_fn(self.temp_path)
            self._move_to_final_location()
        except Exception:
            logger.exception("Error during operation")
            self._cleanup_temp()
            raise

    def _move_to_final_location(self) -> None:
        if self.final_path.exists():
            if self.is_complete():
                if self._uses_custom_validator:
                    logger.info(f"Destination validated as complete: {self.final_path}")
                else:
                    logger.info(f"Destination already exists (created by another process): {self.final_path}")
                self._cleanup_temp()
                return
            raise FileExistsError(f"Destination exists but is incomplete: {self.final_path}")

        logger.info(f"Moving temp to final location: {self.temp_path} -> {self.final_path}")
        try:
            self.temp_path.rename(self.final_path)
            return
        except FileExistsError:
            if self.is_complete():
                if self._uses_custom_validator:
                    logger.info(f"Destination validated as complete during rename: {self.final_path}")
                else:
                    logger.info(f"Destination already exists: {self.final_path}")
                self._cleanup_temp()
                return
            raise
        except OSError:
            logger.warning("Rename failed; validating destination before shutil.move fallback", exc_info=True)

        if self.final_path.exists():
            if self.is_complete():
                if self._uses_custom_validator:
                    logger.info(f"Destination validated as complete after rename race: {self.final_path}")
                else:
                    logger.info(f"Destination created by another process during rename attempt: {self.final_path}")
                self._cleanup_temp()
                return
            raise FileExistsError(f"Destination exists but is incomplete: {self.final_path}")

        try:
            shutil.move(str(self.temp_path), str(self.final_path))
        except FileExistsError:
            if self.is_complete():
                if self._uses_custom_validator:
                    logger.info(f"Destination validated as complete during move: {self.final_path}")
                else:
                    logger.info(f"Destination already exists during move: {self.final_path}")
                self._cleanup_temp()
                return
            raise
        except Exception:
            logger.exception("Move to final location failed")
            raise

        if not self.is_complete():
            raise RuntimeError(f"Destination is incomplete after move: {self.final_path}")

    def _cleanup_temp(self) -> None:
        if self.temp_path.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_path}")
            try:
                shutil.rmtree(self.temp_path)
            except Exception:
                logger.exception("Could not clean up temp directory")


def is_openvino_model_dir(path: Path) -> bool:
    """Return True when path has both OpenVINO XML and BIN model files."""
    if not path.is_dir():
        return False
    has_xml = False
    has_bin = False

    for entry in path.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix == ".xml":
            has_xml = True
        elif entry.suffix == ".bin":
            has_bin = True
        if has_xml and has_bin:
            return True

    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        if entry.suffix == ".xml":
            has_xml = True
        elif entry.suffix == ".bin":
            has_bin = True
        if has_xml and has_bin:
            return True
    return False
