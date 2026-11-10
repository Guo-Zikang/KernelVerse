"""Timestamped progress logging helpers for CLI entrypoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
import threading
from os import PathLike
from typing import TextIO


COUNTED_EVENTS = {
    ("codegen", "start"),
    ("codegen", "done"),
    ("compile", "queued"),
    ("compile", "done"),
    ("eval", "queued"),
    ("eval", "done"),
}


class ProgressLogger:
    """Print timestamped progress events with per-action counters."""

    def __init__(
        self,
        total_ops: int,
        stream: TextIO | None = None,
        log_path: str | PathLike[str] | None = None,
    ) -> None:
        self.total_ops = max(1, int(total_ops))
        self.stream = stream or sys.stdout
        self.log_path = Path(log_path) if log_path is not None else None
        self._lock = threading.Lock()
        self._counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    def _next_index(self, stage: str, action: str) -> int | None:
        if (stage, action) not in COUNTED_EVENTS:
            return None
        key = (stage, action)
        self._counts[key] += 1
        return self._counts[key]

    def log(
        self,
        stage: str,
        action: str,
        *,
        op_name: str | None = None,
        device_id: int | None = None,
        success: bool | None = None,
        elapsed: float | None = None,
        message: str | None = None,
    ) -> None:
        """Print one structured progress event."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            index = self._next_index(stage, action)
            parts = [timestamp, f"[{stage}]", action]
            if index is not None:
                parts.append(f"{index}/{self.total_ops}")
            if op_name is not None:
                parts.append(f"op={op_name}")
            if device_id is not None:
                parts.append(f"device={device_id}")
            if success is not None:
                parts.append(f"success={success}")
            if elapsed is not None:
                parts.append(f"elapsed={elapsed:.1f}s")
            if message:
                parts.append(message)
            line = " ".join(parts)
            print(line, file=self.stream, flush=True)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as log_file:
                    print(line, file=log_file)

    def __call__(self, stage: str, action: str, **kwargs) -> None:
        """Allow instances to be passed directly as callback functions."""
        self.log(stage, action, **kwargs)
