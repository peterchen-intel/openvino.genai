# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3600  # seconds
_LOCK_POLL_INTERVAL = 10  # seconds


class AtomicDownloadManager:
    """Per-path singleton ensuring only one process executes a download at a time.

    State (temp_path, start_time, status) and the lock file are stored under a
    ``.state_<name>`` directory that is a sibling of ``final_path``.  An
    exclusive file lock prevents concurrent execution across processes; other
    callers poll until the lock is released or the timeout expires.
    """

    _instances: dict[Path, "AtomicDownloadManager"] = {}
    _class_lock = threading.Lock()

    def __new__(cls, final_path: Path, timeout: float = _DEFAULT_TIMEOUT) -> "AtomicDownloadManager":
        key = Path(final_path).resolve()
        with cls._class_lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, final_path: Path, timeout: float = _DEFAULT_TIMEOUT) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.final_path = Path(final_path).resolve()
        self._state_dir = self.final_path.parent / f".state_{self.final_path.name}"
        self._lock_file = self._state_dir / "lock"
        self._status_file = self._state_dir / "status.json"
        self._timeout = timeout
        self._lock_fd = None
        self._start_time: float | None = None
        self._instance_lock = threading.Lock()

        random_suffix = uuid.uuid4().hex[:8]
        self.temp_path = self.final_path.parent / f".tmp_{self.final_path.name}_{random_suffix}"

    def is_complete(self) -> bool:
        return self.final_path.exists()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, download_fn: Callable[[Path], None], timeout: float | None = None) -> None:
        if self.is_complete():
            logger.info("Already downloaded: %s", self.final_path)
            return

        effective_timeout = timeout if timeout is not None else self._timeout

        with self._instance_lock:
            if self.is_complete():
                logger.info("Already downloaded: %s", self.final_path)
                return

            if not self._acquire_file_lock(effective_timeout):
                if self.is_complete():
                    logger.info("Completed by another process: %s", self.final_path)
                    return
                raise TimeoutError(f"Timed out waiting for lock on {self.final_path}")

            try:
                if self.is_complete():
                    logger.info("Completed by another process: %s", self.final_path)
                    return

                self._write_status("running")
                self.final_path.parent.mkdir(parents=True, exist_ok=True)
                self.temp_path.mkdir(parents=True, exist_ok=True)

                try:
                    download_fn(self.temp_path)
                    self._move_to_final_location()
                    self._write_status("done")
                except Exception:
                    logger.exception("Error during operation")
                    self._cleanup_temp()
                    self._write_status("failed")
                    raise
            finally:
                self._release_file_lock()

    def force_stop(self) -> None:
        """Force stop: release the lock and remove the temp directory and state."""
        logger.info("Force stopping download for %s", self.final_path)
        self._cleanup_temp()
        self._release_file_lock()
        try:
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception:
            logger.exception("Error removing lock file")
        try:
            if self._state_dir.exists():
                shutil.rmtree(self._state_dir)
        except Exception:
            logger.exception("Error removing state directory")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquire_file_lock(self, timeout: float) -> bool:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            fd = None
            try:
                fd = open(self._lock_file, "w")  # noqa: WPS515
                if _HAS_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_fd = fd
                return True
            except OSError:
                if fd is not None:
                    fd.close()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(_LOCK_POLL_INTERVAL, remaining)
                logger.info("Waiting for lock on %s (%.0fs remaining)...", self.final_path, remaining)
                time.sleep(wait)

    def _release_file_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            if _HAS_FCNTL:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
        except Exception:
            logger.exception("Error releasing lock")
        finally:
            self._lock_fd = None

    def _write_status(self, status: str) -> None:
        if status == "running":
            self._start_time = time.time()
        state = {
            "temp_path": str(self.temp_path),
            "start_time": self._start_time,
            "status": status,
        }
        try:
            with open(self._status_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            logger.exception("Error writing status file")

    def _move_to_final_location(self) -> None:
        if self.final_path.exists():
            logger.info("Destination already exists (created by another process): %s", self.final_path)
            self._cleanup_temp()
            return

        logger.info("Moving temp to final location: %s -> %s", self.temp_path, self.final_path)
        try:
            self.temp_path.rename(self.final_path)
        except Exception:
            logger.warning("Rename failed, falling back to shutil.move")
            if self.final_path.exists():
                logger.info("Destination created by another process during rename attempt: %s", self.final_path)
                self._cleanup_temp()
                return
            try:
                shutil.move(str(self.temp_path), str(self.final_path))
            except Exception:
                logger.exception("Error during move - assuming it was created successfully by another process")
                self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        if self.temp_path.exists():
            logger.info("Cleaning up temp directory: %s", self.temp_path)
            try:
                shutil.rmtree(self.temp_path)
            except Exception:
                logger.exception("Could not clean up temp directory")
