from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = PROJECT_ROOT / "prompts"
MANIFEST_PATH = PROMPT_ROOT / "manifest.json"


@dataclass(frozen=True)
class PromptArtifact:
    artifact_id: str
    version: str
    task: str
    modality: str
    status: str
    path: Path
    text: str
    sha256: str
    variables: tuple[str, ...]


def _read_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported prompt manifest version.")
    return manifest


def resolve_prompt(artifact_id: str, version: str) -> PromptArtifact:
    matches = [
        artifact
        for artifact in _read_manifest()["artifacts"]
        if artifact.get("id") == artifact_id and artifact.get("version") == version
    ]
    if len(matches) != 1:
        raise KeyError(f"Prompt artifact {artifact_id!r} version {version!r} is not uniquely registered.")
    entry = matches[0]
    path = (PROMPT_ROOT / entry["path"]).resolve()
    if not path.is_relative_to(PROMPT_ROOT.resolve()):
        raise ValueError("Prompt artifact path must remain inside the prompt library.")
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != entry.get("sha256"):
        raise ValueError(f"Prompt artifact hash mismatch for {artifact_id!r} version {version!r}.")
    return PromptArtifact(
        artifact_id=artifact_id,
        version=version,
        task=entry["task"],
        modality=entry["modality"],
        status=entry["status"],
        path=path,
        text=text,
        sha256=digest,
        variables=tuple(entry.get("variables", [])),
    )


def render_prompt(artifact: PromptArtifact, **variables: str) -> tuple[str, str]:
    supplied = set(variables)
    expected = set(artifact.variables)
    if supplied != expected:
        raise ValueError(
            f"Prompt variables mismatch; expected {sorted(expected)}, got {sorted(supplied)}."
        )
    rendered = Template(artifact.text).substitute(**variables)
    rendered_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return rendered, rendered_sha256
