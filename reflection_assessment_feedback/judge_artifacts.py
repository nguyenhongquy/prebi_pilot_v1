from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from reflection_assessment_feedback.experiment_config import PROJECT_ROOT


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def protected_path(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise ValueError("Study storage must be outside the source tree.")
    return root


def write_once(path: Path, value: dict) -> None:
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            staged = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        staged.chmod(0o600)
        os.link(staged, path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def private_directory(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root):
        raise ValueError("Study path escapes protected storage.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
