from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.judge_artifacts import (
    content_hash, protected_path as _protected, write_once as _write_once,
    private_directory as _private_directory,
)
from reflection_assessment_feedback.judge_batch import build_judge_batch
from reflection_assessment_feedback.llm_judge import build_judge_input, run_quality_judge
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt
from reflection_assessment_feedback.rate_limit import reserve_request_slot
from reflection_assessment_feedback.reference_inventory import approved_inventory_for_packet
from reflection_assessment_feedback.selected_judge_analysis import (
    expected_judge_metadata, validate_judge_record,
)


def _schedule(entries: list[dict], repetitions: int, seed: int) -> list[dict]:
    schedule = []
    for repetition in range(1, repetitions + 1):
        identifiers = [entry["packet"]["packet_id"] for entry in entries]
        random.Random(seed + repetition).shuffle(identifiers)
        schedule.extend({"packet_id": identifier, "judge_repetition": repetition} for identifier in identifiers)
    return schedule


def _prompt_input(packet: dict, version: str | None) -> dict:
    return build_judge_input(
        packet, assessment_prompt_version=version if packet["task"] == "assessment_quality" else None,
        feedback_prompt_version=version if packet["task"] == "feedback_quality" else None,
    )


def freeze_judge_study(
    protected_root: str | Path,
    *,
    cohort: Literal["playground", "remaining-test"] = "remaining-test",
    conditions: tuple[str, ...] = ("G1", "G2", "G3"),
    judge_repetitions: int = 3,
    import_existing: bool = True,
    order_seed: int = 20261014,
    assessment_prompt_version: str | None = None,
    feedback_prompt_version: str | None = None,
    tasks: tuple[Literal["assessment_quality", "feedback_quality"], ...] = ("assessment_quality", "feedback_quality"),
    source_study: str | Path | None = None,
    document_ids: tuple[str, ...] | None = None,
    reference_inventory_refs: dict[str, dict] | None = None,
) -> Path:
    """Freeze packets, source artifacts, prompts and eligible repetition-1 judgments locally."""
    single_feedback_pilot = feedback_prompt_version in {"0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"} and tasks == ("feedback_quality",)
    if type(judge_repetitions) is not int or judge_repetitions < (1 if single_feedback_pilot else 2):
        raise ValueError("A stability study requires at least two judge repetitions.")
    if type(order_seed) is not int:
        raise ValueError("Order seed must be an integer.")
    root = _protected(protected_root)
    config = load_experiment_config()
    if assessment_prompt_version in {"0.7.0", "0.8.0"} and tasks != ("assessment_quality",):
        raise ValueError("Alignment studies require assessment-quality tasks only.")
    if assessment_prompt_version is not None:
        if "assessment_quality" not in tasks:
            raise ValueError("Assessment prompt override requires assessment-quality tasks.")
        resolve_prompt(config["prompts"]["assessment_quality_llm_id"], assessment_prompt_version)
        if import_existing:
            raise ValueError("A prompt revision requires import_existing=False and fresh judgments.")
    if feedback_prompt_version is not None:
        if tasks != ("feedback_quality",):
            raise ValueError("Feedback alignment studies require feedback-quality tasks only.")
        resolve_prompt(config["prompts"]["feedback_quality_llm_id"], feedback_prompt_version)
        if import_existing:
            raise ValueError("A prompt revision requires import_existing=False and fresh judgments.")
    if reference_inventory_refs is not None and feedback_prompt_version not in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        raise ValueError("Frozen reference inventories require feedback prompt 0.8.0, 0.9.0, 0.10.0, 0.11.0 or 0.12.0.")
    metadata_config = copy.deepcopy(config)
    if assessment_prompt_version is not None:
        metadata_config["prompts"]["assessment_quality_llm_version"] = assessment_prompt_version
    if feedback_prompt_version is not None:
        metadata_config["prompts"]["feedback_quality_llm_version"] = feedback_prompt_version
    original = None
    if source_study is not None:
        original = load_judge_study(source_study)
        if original["protocol"]["cohort"] != cohort:
            raise ValueError("Source study cohort differs from the requested cohort.")
        selected = [
            e for e in original["protocol"]["entries"]
            if e["identity"]["condition"] in conditions and e["packet"]["task"] in tasks
        ]
        packets = [copy.deepcopy(e["packet"]) for e in selected]
        key = {e["packet"]["packet_id"]: copy.deepcopy(e["identity"]) for e in selected}
        sources = copy.deepcopy(original["protocol"]["generation_sources"])
    else:
        packets, key = build_judge_batch(root, cohort=cohort, conditions=conditions, tasks=tasks, pilot_only=False)
        sources = {}
    if document_ids is not None:
        if not document_ids or len(document_ids) != len(set(document_ids)):
            raise ValueError("Document selection must be non-empty and unique.")
        packets = [p for p in packets if key[p["packet_id"]]["document_id"] in document_ids]
        key = {p["packet_id"]: key[p["packet_id"]] for p in packets}
        if {i["document_id"] for i in key.values()} != set(document_ids):
            raise ValueError("Requested documents are missing from the study inputs.")
    if not packets or {(key[p["packet_id"]]["condition"], p["task"]) for p in packets} != {
        (condition, task) for condition in conditions for task in tasks
    }:
        raise ValueError("Requested conditions/tasks are missing from study inputs.")
    source_directory = (("g1-g3", "demo-runs", "comparison-*.json"), ("g2", "demo-runs", "g2-run-*.json"))
    needed = {row["source_artifact_sha256"] for row in key.values()}
    for first, second, pattern in (() if source_study is not None else source_directory):
        for path in (root / first / second).glob(pattern):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in needed:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Generation source escapes protected storage.")
                sources[digest] = path.read_text(encoding="utf-8")
    sources = {digest: text for digest, text in sources.items() if digest in needed}
    if sources.keys() != needed:
        raise ValueError("A generation source changed while freezing; rerun preflight.")
    candidates: dict[str, list[tuple[Path, dict]]] = {p["packet_id"]: [] for p in packets}
    if import_existing:
        for path in sorted((root / "llm-judge" / "development").glob("*/judge-*.json")):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Import source escapes protected storage.")
            record = json.loads(path.read_text())
            if record.get("packet_id") in candidates:
                candidates[record["packet_id"]].append((path, record))
    entries = []
    inventories = {}
    if feedback_prompt_version in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"} and (
        reference_inventory_refs is None or set(reference_inventory_refs) != {
            key[p["packet_id"]]["document_id"] for p in packets
        }
    ):
        raise ValueError("Supply exactly one approved inventory reference per selected reflection.")
    for packet in packets:
        identifier = packet["packet_id"]
        identity = key[identifier]
        packet.pop("reference_inventory", None)
        packet.pop("reference_inventory_ref", None)
        if reference_inventory_refs is not None:
            reference = reference_inventory_refs[identity["document_id"]]
            record = approved_inventory_for_packet(root, reference, packet, identity["document_id"])
            inventories[reference["inventory_id"]] = record
            packet["reference_inventory_ref"] = copy.deepcopy(reference)
            packet["reference_inventory"] = record["inventory"]
        metadata = expected_judge_metadata(packet, identity, metadata_config, cohort)
        prompt = resolve_prompt(metadata["prompt_id"], metadata["prompt_version"])
        rendered, rendered_hash = render_prompt(
            prompt, packet_json=json.dumps(
                _prompt_input(packet, prompt.version), ensure_ascii=False, sort_keys=True, indent=2,
            ),
        )
        metadata.update({"rendered_prompt_sha256": rendered_hash, "judge_parameters": {}})
        matches = candidates[identifier]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple import candidates for packet {identifier}; resolve explicitly or freeze with import_existing=False."
            )
        imported = None
        if matches:
            path, record = matches[0]
            validate_judge_record(record, packet, metadata)
            if not record.get("created_at") or not record.get("run_id"):
                raise ValueError("Imported judgment must retain its original timestamp and run ID.")
            imported = {
                "source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "record": record,
            }
        entries.append({
            "packet": {k: v for k, v in packet.items() if k != "reference_inventory"}, "identity": identity,
            "generation_repetition": identity["repetition"],
            "input_sha256": content_hash(_prompt_input(packet, prompt.version)),
            "expected_metadata": metadata, "prompt_text": prompt.text,
            "rendered_prompt": rendered, "imported_judgment": imported,
        })
    protocol = {
        "schema_version": "1.0.0", "cohort": cohort, "conditions": list(conditions),
        "judge_repetitions": judge_repetitions, "order_seed": order_seed,
        "judge_config": config["llm_judge"], "prompt_config": config["prompts"],
        "experiment_config": config, "generation_sources": sources, "entries": entries,
        "schedule": _schedule(entries, judge_repetitions, order_seed),
        "analysis_policy": {
            "unit": "document/generation-repetition/task/criterion",
            "complete_repetitions_required": True,
            "unable_to_judge": "report separately; exclude from numerical comparisons",
            "duplicate_repetition": "error", "extension": "separate prospective decision",
        },
    }
    if assessment_prompt_version is not None:
        protocol["assessment_prompt_version"] = assessment_prompt_version
    if feedback_prompt_version is not None:
        protocol["feedback_prompt_version"] = feedback_prompt_version
    if inventories:
        protocol["reference_inventories"] = inventories
    if original is not None:
        protocol["source_study_id"] = original["study_id"]
    if document_ids is not None:
        protocol["document_ids"] = list(document_ids)
    study_id = content_hash(protocol)
    directory = root / "llm-judge" / "stability-studies" / study_id
    _private_directory(directory, root)
    manifest = directory / "manifest.json"
    if manifest.exists():
        load_judge_study(directory)
    else:
        _write_once(manifest, {"study_id": study_id, "protocol": protocol})
    _import_frozen_results(directory)
    return directory


def evaluation_packet(protocol: dict, entry: dict) -> dict:
    packet = copy.deepcopy(entry["packet"])
    if protocol.get("feedback_prompt_version") in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        reference = packet["reference_inventory_ref"]
        record = protocol["reference_inventories"][reference["inventory_id"]]
        packet["reference_inventory"] = copy.deepcopy(record["inventory"])
    return packet


def load_judge_study(directory: str | Path) -> dict:
    directory = _protected(directory)
    manifest = directory / "manifest.json"
    if not manifest.resolve().is_relative_to(directory):
        raise ValueError("Manifest escapes study storage.")
    study = json.loads(manifest.read_text())
    protocol = study["protocol"]
    if study["study_id"] != content_hash(protocol) or directory.name != study["study_id"]:
        raise ValueError("Frozen study manifest hash mismatch.")
    if protocol["schema_version"] != "1.0.0":
        raise ValueError("Unsupported study schema.")
    for digest, text in protocol["generation_sources"].items():
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise ValueError("Frozen generation source hash mismatch.")
    for entry in protocol["entries"]:
        packet = evaluation_packet(protocol, entry)
        if protocol.get("feedback_prompt_version") in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
            reference = packet["reference_inventory_ref"]
            approved = approved_inventory_for_packet(
                directory.parents[2], reference, packet, entry["identity"]["document_id"],
            )
            if approved != protocol["reference_inventories"][reference["inventory_id"]]:
                raise ValueError("Frozen inventory artifact differs from approved source.")
        if content_hash(_prompt_input(
            packet, entry.get("expected_metadata", {}).get(
                "prompt_version", protocol.get("assessment_prompt_version"),
            ),
        )) != entry["input_sha256"]:
            raise ValueError("Frozen input hash mismatch.")
    return study


def _result_path(directory: Path, identifier: str, repetition: int) -> Path:
    path = directory / "results" / f"judge-{UUID(identifier).hex}-rep-{repetition}.json"
    if not path.resolve().is_relative_to(directory):
        raise ValueError("Result escapes study storage.")
    return path


def _validate_study_result(study: dict, entry: dict, repetition: int, record: dict):
    if (
        record.get("study_id") != study["study_id"]
        or record.get("judge_repetition") != repetition
        or record.get("generation_repetition") != entry["generation_repetition"]
        or record.get("frozen_input_sha256") != entry["input_sha256"]
    ):
        raise ValueError("Study result repetition or frozen provenance mismatch.")
    imported = entry["imported_judgment"] if repetition == 1 else None
    if imported is not None:
        if record.get("origin") != "imported" or record["judgment"] != imported["record"]:
            raise ValueError("Imported repetition differs from its frozen original judgment.")
    elif record.get("origin") != "new":
        raise ValueError("Only the frozen repetition-1 judgment may be imported.")
    return validate_judge_record(
        record["judgment"], evaluation_packet(study["protocol"], entry), entry["expected_metadata"],
    )


def _study_record(study: dict, entry: dict, repetition: int, judgment: dict, imported: dict | None) -> dict:
    return {
        "study_id": study["study_id"], "judge_repetition": repetition,
        "generation_repetition": entry["generation_repetition"],
        "frozen_input_sha256": entry["input_sha256"],
        "origin": "imported" if imported else "new",
        "import_source": (
            {k: imported[k] for k in ("source_path", "source_sha256")} if imported else None
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(), "judgment": judgment,
    }


def _import_frozen_results(directory: Path) -> None:
    study = load_judge_study(directory)
    _private_directory(directory / "results", directory)
    with (directory / ".study.lock").open("a+") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, existing = study_results(directory)
        for entry in study["protocol"]["entries"]:
            imported = entry["imported_judgment"]
            identifier = entry["packet"]["packet_id"]
            if imported is not None and (identifier, 1) not in existing:
                record = _study_record(study, entry, 1, imported["record"], imported)
                _validate_study_result(study, entry, 1, record)
                _write_once(_result_path(directory, identifier, 1), record)


def study_results(directory: str | Path) -> tuple[dict, dict]:
    directory = _protected(directory)
    study = load_judge_study(directory)
    entries = {e["packet"]["packet_id"]: e for e in study["protocol"]["entries"]}
    results = {}
    for path in sorted((directory / "results").glob("*.json")):
        record = json.loads(path.read_text())
        identifier = record.get("judgment", {}).get("packet_id")
        repetition = record.get("judge_repetition")
        if (
            identifier not in entries or type(repetition) is not int
            or not 1 <= repetition <= study["protocol"]["judge_repetitions"]
        ):
            raise ValueError("Unexpected study result or repetition.")
        if path != _result_path(directory, identifier, repetition):
            raise ValueError("Duplicate or noncanonical study result filename.")
        _validate_study_result(study, entries[identifier], repetition, record)
        results[(identifier, repetition)] = record
    return study, results


def study_progress(directory: str | Path) -> dict:
    study, results = study_results(directory)
    entries = study["protocol"]["entries"]
    imported = sum(e["imported_judgment"] is not None for e in entries)
    pending_imports = sum(
        e["imported_judgment"] is not None and (e["packet"]["packet_id"], 1) not in results
        for e in entries
    )
    total = len(entries) * study["protocol"]["judge_repetitions"]
    return {
        "study_id": study["study_id"], "cohort": study["protocol"]["cohort"],
        "conditions": study["protocol"]["conditions"],
        "packets": len(entries), "judge_repetitions": study["protocol"]["judge_repetitions"],
        "planned_judgments": total, "eligible_imports": imported, "saved_judgments": len(results),
        "pending_imports": pending_imports,
        "maximum_new_calls_without_retries": total - len(results) - pending_imports,
    }


def run_judge_study(
    directory: str | Path, *, approve_external_processing: bool = False,
    approve_full_batch: bool = False, model=None,
) -> dict:
    if not approve_external_processing or not approve_full_batch:
        raise RuntimeError("Explicit external-processing and full-study approval are required.")
    directory = _protected(directory)
    study = load_judge_study(directory)
    protocol = study["protocol"]
    current = load_experiment_config()
    if current["llm_judge"] != protocol["judge_config"] or current["prompts"] != protocol["prompt_config"]:
        raise ValueError("Judge configuration changed since freezing; do not mix study protocols.")
    for entry in protocol["entries"]:
        prompt = resolve_prompt(entry["expected_metadata"]["prompt_id"], entry["expected_metadata"]["prompt_version"])
        if prompt.text != entry["prompt_text"]:
            raise ValueError("Judge prompt changed since freezing.")
    if model is None:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=protocol["judge_config"]["model_name"])
    if getattr(model, "model_name", protocol["judge_config"]["model_name"]) != protocol["judge_config"]["model_name"]:
        raise ValueError("Supplied model does not match the frozen judge model.")
    parameters = {
        name: value for name in ("top_p", "max_tokens", "seed")
        if (value := getattr(model, name, None)) is not None
    }
    if parameters != {} or getattr(model, "temperature", None) is not None:
        raise ValueError("Study requires the frozen unset sampling/parameter configuration.")
    _private_directory(directory / "results", directory)
    entries = {e["packet"]["packet_id"]: e for e in protocol["entries"]}
    with (directory / ".study.lock").open("a+") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, existing = study_results(directory)
        for slot in protocol["schedule"]:
            identifier, repetition = slot["packet_id"], slot["judge_repetition"]
            if (identifier, repetition) in existing:
                continue
            entry = entries[identifier]
            imported = entry["imported_judgment"] if repetition == 1 else None
            if imported is not None:
                judgment = imported["record"]
            else:
                reserve_request_slot(
                    directory.parents[2] / ".rate_limits" / "openai-judge.state",
                    protocol["judge_config"]["minimum_request_interval_seconds"],
                )
                judgment = run_quality_judge(
                    model=model, packet=evaluation_packet(protocol, entry),
                    provider=protocol["judge_config"]["provider"],
                    model_name=protocol["judge_config"]["model_name"],
                    enable_langsmith_tracing=protocol["judge_config"]["enable_langsmith_tracing"],
                    **(
                        {"assessment_prompt_version": protocol["assessment_prompt_version"]}
                        if entry["packet"]["task"] == "assessment_quality" and "assessment_prompt_version" in protocol
                        else {}
                    ),
                    **(
                        {"feedback_prompt_version": protocol["feedback_prompt_version"]}
                        if entry["packet"]["task"] == "feedback_quality" and "feedback_prompt_version" in protocol
                        else {}
                    ),
                )
                expected = entry["expected_metadata"]
                if any(name in judgment and judgment[name] != value for name, value in expected.items()):
                    raise ValueError("New judgment conflicts with frozen provenance.")
                judgment = {**judgment, **expected}
            record = _study_record(study, entry, repetition, judgment, imported)
            _validate_study_result(study, entry, repetition, record)
            _write_once(_result_path(directory, identifier, repetition), record)
    return study_progress(directory)
