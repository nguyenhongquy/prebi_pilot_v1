from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflection_assessment_feedback.judge_artifacts import (
    content_hash, private_directory, protected_path, write_once,
)
from reflection_assessment_feedback.judge_provider import invoke_judge_model
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt
from reflection_assessment_feedback.rate_limit import reserve_request_slot


class InventoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FixedReferenceConcern(InventoryModel):
    concern_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    concern_type: Literal["positive", "critical", "developmental"]
    human_excerpt: str = Field(min_length=1)
    normalized_concern: str = Field(min_length=1)
    centrality: Literal["central", "secondary"]
    centrality_rationale: str = Field(min_length=1)
    reference_support: Literal["supported", "uncertain", "unsupported"]
    evidence_segment_ids: list[str]
    rationale: str = Field(min_length=1)


class ReferenceConcernInventory(InventoryModel):
    schema_version: Literal["1.0.0"]
    reflection_id: str = Field(min_length=1)
    reference_concerns: list[FixedReferenceConcern]

    @model_validator(mode="after")
    def validate_identifiers(self):
        if [c.concern_id for c in self.reference_concerns] != [
            f"R{i}" for i in range(1, len(self.reference_concerns) + 1)
        ]:
            raise ValueError("Inventory IDs must be consecutive R1, R2, ... in reference order.")
        return self


class EvaluableReferenceConcern(FixedReferenceConcern):
    evaluability: Literal["evaluable", "partially_evaluable", "not_evaluable"]
    evaluability_rationale: str = Field(min_length=1)
    evaluable_scope: str | None
    unavailable_context: list[str]

    @model_validator(mode="after")
    def validate_scope(self):
        if self.evaluability == "not_evaluable":
            if self.evaluable_scope is not None or not self.unavailable_context:
                raise ValueError("Non-evaluable concerns require missing context and no evaluable scope.")
        elif not self.evaluable_scope or not self.evaluable_scope.strip():
            raise ValueError("Evaluable concerns require an explicit evaluable scope.")
        if self.evaluability == "partially_evaluable" and not self.unavailable_context:
            raise ValueError("Partially evaluable concerns require unavailable context.")
        if any(not item.strip() for item in self.unavailable_context):
            raise ValueError("Unavailable context descriptions must not be empty.")
        return self


class EvaluableConcernInventory(ReferenceConcernInventory):
    schema_version: Literal["1.1.0"]
    reference_concerns: list[EvaluableReferenceConcern]


def parse_inventory(data: dict) -> ReferenceConcernInventory | EvaluableConcernInventory:
    model = EvaluableConcernInventory if data.get("schema_version") == "1.1.0" else ReferenceConcernInventory
    return model.model_validate(data)


class ExtractorConfig(InventoryModel):
    prompt_id: str = "feedback_reference_concern_extractor"
    prompt_version: str = "0.1.0"
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    extraction_revision: str = Field(default="1", min_length=1)

    @model_validator(mode="after")
    def validate_parameters(self):
        if set(self.parameters) - {"temperature", "top_p", "max_tokens", "seed"}:
            raise ValueError("Extractor parameters must be explicit supported sampling parameters.")
        if any(value is None for value in self.parameters.values()):
            raise ValueError("Omit unset extractor parameters rather than recording null values.")
        return self


def reference_input(
    *, reflection_id: str, reflection: dict, reference_teacher_feedback: str,
    rubric: dict, rubric_version: str,
) -> dict:
    if not reflection_id or not reference_teacher_feedback.strip():
        raise ValueError("Reference extraction requires a reflection ID and lecturer feedback.")
    if rubric.get("version") != rubric_version:
        raise ValueError("Rubric version differs from its source artifact.")
    segments = [
        {key: s[key] for key in ("segment_id", "order", "text") if key in s}
        for s in reflection["segments"]
    ]
    if not segments or any(not s.get("segment_id") or not s.get("text") for s in segments):
        raise ValueError("Reference extraction requires full reflection segments with IDs.")
    if len({s["segment_id"] for s in segments}) != len(segments):
        raise ValueError("Reflection segment IDs must be unique.")
    return {
        "reflection_id": reflection_id, "reflection": {"segments": segments},
        "reference_teacher_feedback": reference_teacher_feedback,
        "rubric": rubric, "rubric_version": rubric_version,
    }


def reference_input_from_packet(packet: dict, reflection_id: str) -> dict:
    return reference_input(
        reflection_id=reflection_id, reflection=packet["reflection"],
        reference_teacher_feedback=packet["reference_teacher_feedback"],
        rubric=packet["rubric"], rubric_version=packet["rubric_version"],
    )


def validate_inventory(inventory: ReferenceConcernInventory, source: dict) -> None:
    if inventory.reflection_id != source["reflection_id"]:
        raise ValueError("Inventory reflection ID differs from its source.")
    text = " ".join(source["reference_teacher_feedback"].split())
    segments = {s["segment_id"] for s in source["reflection"]["segments"]}
    previous = -1
    for concern in inventory.reference_concerns:
        excerpt = " ".join(concern.human_excerpt.split())
        position = text.find(excerpt) if excerpt else -1
        if position < 0:
            raise ValueError("Inventory human excerpt is not present in lecturer feedback.")
        if position < previous:
            raise ValueError("Inventory concerns must follow lecturer-feedback order.")
        previous = position
        if not set(concern.evidence_segment_ids).issubset(segments):
            raise ValueError("Inventory cites an unknown reflection segment.")


def inventory_specification(source: dict, config: ExtractorConfig) -> dict:
    artifact = resolve_prompt(config.prompt_id, config.prompt_version)
    if artifact.task != "reference_concern_extraction" or artifact.status != "development_draft":
        raise ValueError("Extractor prompt has an incompatible task or status.")
    return {
        "schema_version": "1.0.0", "source": source,
        "extractor": config.model_dump(mode="json"),
        "prompt_artifact_sha256": artifact.sha256,
    }


def _directory(root: Path, inventory_id: str) -> Path:
    if len(inventory_id) != 64 or any(c not in "0123456789abcdef" for c in inventory_id):
        raise ValueError("Inventory ID must be a canonical SHA256.")
    directory = root / "llm-judge" / "reference-inventories" / inventory_id
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Inventory escapes protected storage.")
    return directory


def load_inventory_artifact(directory: str | Path, *, require_approved: bool = False) -> dict:
    directory = protected_path(directory)
    path = directory / "draft.json"
    if not path.resolve().is_relative_to(directory):
        raise ValueError("Inventory artifact escapes storage.")
    record = json.loads(path.read_text())
    spec = record["specification"]
    if record["inventory_id"] != directory.name or content_hash(spec) != record["inventory_id"]:
        raise ValueError("Inventory specification hash mismatch.")
    source = spec["source"]
    if source != reference_input(**source):
        raise ValueError("Inventory source contains unapproved fields.")
    if record["source_hashes"] != {key: content_hash(value) for key, value in source.items()}:
        raise ValueError("Inventory source hash mismatch.")
    inventory = parse_inventory(record["inventory"])
    validate_inventory(inventory, source)
    if content_hash(record["inventory"]) != record["inventory_sha256"]:
        raise ValueError("Inventory content hash mismatch.")
    config = ExtractorConfig.model_validate(spec["extractor"])
    expected_schema = "1.1.0" if config.prompt_version == "0.2.0" else "1.0.0"
    if inventory.schema_version != expected_schema:
        raise ValueError("Inventory schema differs from extractor version.")
    if inventory_specification(source, config) != spec:
        raise ValueError("Extractor prompt provenance mismatch.")
    prompt = resolve_prompt(config.prompt_id, config.prompt_version)
    _, digest = render_prompt(prompt, packet_json=json.dumps(
        source, ensure_ascii=False, sort_keys=True, indent=2,
    ))
    if record["rendered_prompt_sha256"] != digest or record["prompt_text"] != prompt.text:
        raise ValueError("Extractor rendered prompt hash mismatch.")
    if record["schema_version"] != "1.0.0" or not record["created_at"] or not record["run_id"]:
        raise ValueError("Inventory creation provenance is missing.")
    if require_approved:
        approval_path = directory / "approval.json"
        if not approval_path.resolve().is_relative_to(directory):
            raise ValueError("Inventory approval escapes storage.")
        if not approval_path.exists():
            raise ValueError("Draft inventory requires explicit approval before batch freezing.")
        approval = json.loads(approval_path.read_text())
        expected = {
            "schema_version": "1.0.0", **inventory_reference(record),
            "draft_sha256": content_hash(record),
        }
        if any(approval.get(k) != v for k, v in expected.items()) or not (
            approval.get("approved_by", "").strip() and approval.get("approved_at")
        ):
            raise ValueError("Inventory approval or finalized artifact hash mismatch.")
    return record


def inventory_reference(record: dict) -> dict:
    return {
        "inventory_id": record["inventory_id"], "inventory_sha256": record["inventory_sha256"],
        "reflection_id": record["inventory"]["reflection_id"],
    }


def approve_inventory(directory: str | Path, *, approved_by: str) -> dict:
    directory = protected_path(directory)
    if not approved_by.strip():
        raise ValueError("Inventory approval requires an identified reviewer.")
    with (directory / ".inventory.lock").open("a+") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = load_inventory_artifact(directory)
        approval_path = directory / "approval.json"
        if not approval_path.exists():
            write_once(approval_path, {
                "schema_version": "1.0.0", **inventory_reference(record),
                "draft_sha256": content_hash(record), "approved_by": approved_by.strip(),
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })
        load_inventory_artifact(directory, require_approved=True)
        return inventory_reference(record)


def extract_reference_inventory(
    protected_root: str | Path, *, source: dict, config: ExtractorConfig,
    model=None, approve_external_processing: bool = False,
    enable_langsmith_tracing: bool = False, minimum_request_interval_seconds: float = 1,
) -> Path:
    source = reference_input(**source)
    spec = inventory_specification(source, config)
    root = protected_path(protected_root)
    directory = _directory(root, content_hash(spec))
    private_directory(directory, root)
    with (directory / ".inventory.lock").open("a+") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "draft.json").exists():
            load_inventory_artifact(directory)
            return directory
        if not approve_external_processing:
            raise RuntimeError("External-processing approval is required for reference extraction.")
        if config.provider != "openai" and model is None:
            raise ValueError("Default reference extractor supports the configured OpenAI judge provider.")
        if model is None:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=config.model_name, **config.parameters)
        actual = {
            name: value for name in ("temperature", "top_p", "max_tokens", "seed")
            if (value := getattr(model, name, None)) is not None
        }
        if actual != config.parameters or getattr(model, "model_name", config.model_name) != config.model_name:
            raise ValueError("Extractor model or parameters differ from the cache specification.")
        prompt = resolve_prompt(config.prompt_id, config.prompt_version)
        rendered, digest = render_prompt(prompt, packet_json=json.dumps(
            source, ensure_ascii=False, sort_keys=True, indent=2,
        ))
        run_id = str(uuid4())
        reserve_request_slot(
            root / ".rate_limits" / "openai-judge.state",
            minimum_request_interval_seconds,
        )
        output_model = EvaluableConcernInventory if config.prompt_version == "0.2.0" else ReferenceConcernInventory
        raw = invoke_judge_model(
            model=model, schema=output_model, prompt=rendered,
            enable_langsmith_tracing=enable_langsmith_tracing,
            metadata={
                "task": "reference_concern_extraction", "run_id": run_id,
                "inventory_id": directory.name, "prompt_id": prompt.artifact_id,
                "prompt_version": prompt.version, "rendered_prompt_sha256": digest,
            },
        )
        inventory = output_model.model_validate(raw)
        validate_inventory(inventory, source)
        record = {
            "schema_version": "1.0.0", "inventory_id": directory.name, "specification": spec,
            "source_hashes": {key: content_hash(value) for key, value in source.items()},
            "inventory": inventory.model_dump(mode="json"),
            "inventory_sha256": content_hash(inventory.model_dump(mode="json")),
            "created_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
            "prompt_text": prompt.text, "rendered_prompt_sha256": digest,
        }
        write_once(directory / "draft.json", record)
    load_inventory_artifact(directory)
    return directory


def approved_inventory_for_packet(
    protected_root: str | Path, reference: dict, packet: dict, reflection_id: str,
) -> dict:
    root = protected_path(protected_root)
    record = load_inventory_artifact(
        _directory(root, reference["inventory_id"]), require_approved=True,
    )
    if inventory_reference(record) != reference:
        raise ValueError("Frozen inventory reference differs from approved inventory.")
    if record["specification"]["source"] != reference_input_from_packet(packet, reflection_id):
        raise ValueError("Inventory source differs from evaluation packet.")
    return record
