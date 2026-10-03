from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflection_assessment_feedback.models import Rubric

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_ROOT = PROJECT_ROOT / "rubrics"
MANIFEST_PATH = RUBRIC_ROOT / "manifest.json"


@dataclass(frozen=True)
class RubricArtifact:
    rubric_id: str
    version: str
    status: str
    path: Path
    rubric: Rubric
    sha256: str


def _read_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported rubric manifest version.")
    return manifest


def resolve_rubric(rubric_id: str, version: str) -> RubricArtifact:
    matches = [
        item
        for item in _read_manifest()["artifacts"]
        if item.get("id") == rubric_id and item.get("version") == version
    ]
    if len(matches) != 1:
        raise KeyError(f"Rubric {rubric_id!r} version {version!r} is not uniquely registered.")
    entry = matches[0]
    path = (RUBRIC_ROOT / entry["path"]).resolve()
    if not path.is_relative_to(RUBRIC_ROOT.resolve()):
        raise ValueError("Rubric artifact path must remain inside the rubric library.")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry.get("sha256"):
        raise ValueError(f"Rubric hash mismatch for {rubric_id!r} version {version!r}.")
    rubric = Rubric.model_validate(json.loads(content))
    if rubric.rubric_id != rubric_id or rubric.version != version:
        raise ValueError("Rubric file identity does not match its manifest entry.")
    return RubricArtifact(
        rubric_id=rubric_id,
        version=version,
        status=entry["status"],
        path=path,
        rubric=rubric,
        sha256=digest,
    )
