from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path


def reserve_request_slot(state_path: str | Path, minimum_interval_seconds: float) -> float:
    if minimum_interval_seconds <= 0:
        raise ValueError("minimum_interval_seconds must be positive.")
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path.parent.chmod(0o700)

    waited_seconds = 0.0
    with state_path.open("a+", encoding="ascii") as state_file:
        os.chmod(state_path, 0o600)
        fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
        try:
            state_file.seek(0)
            previous_value = state_file.read().strip()
            previous_request_time = float(previous_value) if previous_value else None
            now = time.time()
            if previous_request_time is not None:
                waited_seconds = max(
                    0.0,
                    minimum_interval_seconds - (now - previous_request_time),
                )
                if waited_seconds:
                    time.sleep(waited_seconds)
            reserved_at = time.time()
            state_file.seek(0)
            state_file.truncate()
            state_file.write(f"{reserved_at:.6f}\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        finally:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)
    return waited_seconds
