"""Seal a selection of completed feedback-judge results as an immutable result set.

The freeze manifest records hashes of the study manifests, judge result files and the
embedded generation artifacts, plus the micro-averaged metrics computed from them. It
never calls a provider and never modifies study data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from reflection_assessment_feedback.judge_artifacts import (
    content_hash, private_directory, protected_path, write_once,
)
from reflection_assessment_feedback.judge_stability import (
    extract_feedback_measurements, summarize_micro_feedback_metrics,
)
from reflection_assessment_feedback.judge_study import _result_path, study_results

SCHEMA_VERSION = "1.0.0"
GENERATORS = ("G1", "G2", "G3")


def _find_run(value, run_id: str) -> list[dict]:
    if isinstance(value, dict):
        found = [value] if value.get("run_id") == run_id and "prompt_version" in value else []
        return found + [run for child in value.values() for run in _find_run(child, run_id)]
    if isinstance(value, list):
        return [run for child in value for run in _find_run(child, run_id)]
    return []


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_feedback_result_freeze(
    root: str | Path, *, label: str, selections: dict[str, tuple[str, ...]],
    generation_prompt_version: str, judge_prompt_version: str, judge_repetition: int = 1,
) -> dict:
    """Validate the selected study results and return the freeze record (no writes)."""
    root = protected_path(root)
    selected = [condition for conditions in selections.values() for condition in conditions]
    if sorted(selected) != sorted(GENERATORS):
        raise ValueError("Select each of G1, G2 and G3 from exactly one study.")
    studies, items, measurements, protocols = [], [], [], []
    for study_id, conditions in selections.items():
        directory = root / "llm-judge" / "stability-studies" / study_id
        study, results = study_results(directory)
        protocol = study["protocol"]
        protocols.append(protocol)
        if protocol.get("feedback_prompt_version") != judge_prompt_version:
            raise ValueError(f"Study {study_id[:12]} does not use judge prompt {judge_prompt_version}.")
        if not 1 <= judge_repetition <= protocol["judge_repetitions"]:
            raise ValueError(f"Study {study_id[:12]} has no judge repetition {judge_repetition}.")
        if not set(conditions) <= set(protocol["conditions"]):
            raise ValueError(f"Study {study_id[:12]} lacks a selected generator.")
        studies.append({
            "study_id": study_id, "generators": list(conditions),
            "manifest_sha256": _file_sha256(directory / "manifest.json"),
        })
        for entry in protocol["entries"]:
            identity = entry["identity"]
            if identity["condition"] not in conditions:
                continue
            identifier = entry["packet"]["packet_id"]
            if (identifier, judge_repetition) not in results:
                raise ValueError(f"Study {study_id[:12]} has an unsaved selected judgment.")
            source = json.loads(protocol["generation_sources"][identity["source_artifact_sha256"]])
            runs = _find_run(source, identity["run_id"])
            if len(runs) != 1:
                raise ValueError("Selected run is not uniquely present in its frozen generation source.")
            run = runs[0]
            components = [
                {key: component[key] for key in ("artifact_id", "version", "sha256")}
                for component in run.get("prompt_components") or []
            ]
            if (
                run["condition"] != identity["condition"]
                or run["prompt_version"] != generation_prompt_version
                or not components
                or any(component["version"] != generation_prompt_version for component in components)
            ):
                raise ValueError(
                    f"{identity['condition']} run for document {identity['document_id']} "
                    f"is not generation prompt {generation_prompt_version}.",
                )
            result_path = _result_path(directory, identifier, judge_repetition)
            reference = entry["packet"]["reference_inventory_ref"]
            items.append({
                "generator": identity["condition"], "document_id": identity["document_id"],
                "study_id": study_id, "packet_id": identifier,
                "generation_repetition": entry["generation_repetition"],
                "judge_repetition": judge_repetition,
                "run_id": identity["run_id"],
                "source_artifact_sha256": identity["source_artifact_sha256"],
                "generation": {
                    key: run[key] for key in (
                        "model_name", "prompt_version", "pipeline_version", "rubric_version",
                        "rubric_artifact_sha256", "rendered_prompt_sha256",
                    )
                } | {"prompt_components": components},
                "frozen_input_sha256": entry["input_sha256"],
                "inventory_id": reference["inventory_id"],
                "inventory_sha256": reference["inventory_sha256"],
                "result_file": str(result_path.relative_to(root)),
                "result_sha256": _file_sha256(result_path),
            })
        table = extract_feedback_measurements(directory)["measurements"]
        measurements.append(table[
            table["condition"].isin(conditions) & table["judge_repetition"].eq(judge_repetition)
        ])
    if len({p["cohort"] for p in protocols}) != 1:
        raise ValueError("Selected studies belong to different cohorts.")
    if len({json.dumps(p["judge_config"], sort_keys=True) for p in protocols}) != 1:
        raise ValueError("Selected studies use different judge configurations.")
    documents = {g: {i["document_id"] for i in items if i["generator"] == g} for g in GENERATORS}
    if len({frozenset(d) for d in documents.values()}) != 1:
        raise ValueError("G1, G2 and G3 must cover the same reflections.")
    inventories = {}
    for item in items:
        inventories.setdefault(item["document_id"], set()).add((item["inventory_id"], item["inventory_sha256"]))
    if any(len(refs) != 1 for refs in inventories.values()):
        raise ValueError("Generators were judged against different inventories for the same reflection.")
    if len({(i["generator"], i["document_id"]) for i in items}) != len(items):
        raise ValueError("Each generator must have one judged feedback per reflection.")
    metrics = summarize_micro_feedback_metrics(pd.concat(measurements, ignore_index=True))
    judge = protocols[0]["judge_config"]
    record = {
        "schema_version": SCHEMA_VERSION, "label": label, "cohort": protocols[0]["cohort"],
        "generation_prompt_version": generation_prompt_version,
        "judge_prompt_version": judge_prompt_version, "judge_repetition": judge_repetition,
        "judge": {"provider": judge["provider"], "model_name": judge["model_name"]},
        "document_ids": sorted(documents["G1"], key=lambda d: (len(d), d)),
        "studies": studies,
        "items": sorted(items, key=lambda i: (i["generator"], len(i["document_id"]), i["document_id"])),
        "metric_definition": {
            "aggregation": "micro: counts summed across reflections before division; repetitions separate",
            "PRR": "fully addressed / eligible positive concerns",
            "CMR_CPCR_CFCR": "missing / partially / fully addressed over eligible central critical/developmental concerns; sum to 100%",
            "MDI": "feedbacks with >=1 judge-classified major Issue / evaluated feedbacks",
            "exclusions": "unsupported, not_evaluable and unable_to_judge concerns; zero denominators are null",
            "units": "percent, rounded to two decimals",
        },
        "metrics": json.loads(metrics.to_json(orient="records")),
    }
    return {"freeze_id": content_hash(record), "record": record}


def freeze_feedback_results(root: str | Path, **kwargs) -> Path:
    """Write the freeze manifest once; an identical existing freeze is accepted."""
    root = protected_path(root)
    frozen = build_feedback_result_freeze(root, **kwargs)
    directory = root / "llm-judge" / "result-freezes" / frozen["freeze_id"]
    private_directory(directory, root)
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Existing freeze manifest differs from the recomputed freeze.")
    else:
        write_once(path, frozen)
    return directory


def verify_feedback_result_freeze(root: str | Path, freeze_id: str) -> dict:
    """Recompute the freeze from current study data and require an exact match."""
    root = protected_path(root)
    path = root / "llm-judge" / "result-freezes" / freeze_id / "manifest.json"
    saved = json.loads(path.read_text())
    record = saved["record"]
    if saved["freeze_id"] != freeze_id or content_hash(record) != freeze_id:
        raise ValueError("Freeze manifest hash mismatch.")
    recomputed = build_feedback_result_freeze(
        root, label=record["label"],
        selections={s["study_id"]: tuple(s["generators"]) for s in record["studies"]},
        generation_prompt_version=record["generation_prompt_version"],
        judge_prompt_version=record["judge_prompt_version"],
        judge_repetition=record["judge_repetition"],
    )
    if recomputed != saved:
        raise ValueError("Frozen results changed since freezing.")
    return saved


def _selection(text: str) -> tuple[str, tuple[str, ...]]:
    study_id, _, generators = text.partition("=")
    if not study_id or not generators:
        raise argparse.ArgumentTypeError("Use STUDY_ID=G1,G3.")
    return study_id, tuple(g.strip() for g in generators.split(","))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-root", default=os.environ.get("PREBI_DATA_ROOT"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--select", type=_selection, action="append", required=True)
    parser.add_argument("--generation-prompt-version", required=True)
    parser.add_argument("--judge-prompt-version", required=True)
    parser.add_argument("--judge-repetition", type=int, default=1)
    parser.add_argument("--write", action="store_true", help="Write the freeze manifest (default: preview).")
    args = parser.parse_args(argv)
    if not args.protected_root:
        parser.error("Set PREBI_DATA_ROOT or --protected-root.")
    options = {
        "label": args.label, "selections": dict(args.select),
        "generation_prompt_version": args.generation_prompt_version,
        "judge_prompt_version": args.judge_prompt_version, "judge_repetition": args.judge_repetition,
    }
    if args.write:
        directory = freeze_feedback_results(args.protected_root, **options)
        print(f"Frozen: {directory}")
        frozen = verify_feedback_result_freeze(args.protected_root, directory.name)
    else:
        frozen = build_feedback_result_freeze(args.protected_root, **options)
        print(f"Preview only; freeze id would be {frozen['freeze_id']}")
    print(pd.DataFrame(frozen["record"]["metrics"])[
        ["Generator", "generation_repetition", "judge_repetition", "PRR", "CMR", "CPCR", "CFCR", "MDI"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
