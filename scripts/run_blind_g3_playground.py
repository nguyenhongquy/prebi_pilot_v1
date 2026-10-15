"""Reference-blind judging of seven existing document-3 G3 assessments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

from reflection_assessment_feedback.generator import validate_output
from reflection_assessment_feedback.judge_stability import analyze_alignment_study
from reflection_assessment_feedback.judge_study import (
    _private_directory, _protected, _write_once, content_hash, freeze_judge_study,
    run_judge_study, study_progress, study_results,
)
from reflection_assessment_feedback.llm_judge import (
    AssessmentAlignmentJudgeOutput, build_judge_input, judge_input_sha256,
    validate_alignment_evidence,
)
from reflection_assessment_feedback.models import DocumentGenerationOutput, GenerationRequest
from scripts.regenerate_g3_playground import assessment_packet, prepare as prepare_generation


def prepare(root: Path, source_directory: Path) -> Path:
    root = _protected(root)
    original, results = study_results(source_directory)
    if original["protocol"].get("assessment_prompt_version") != "0.7.0":
        raise ValueError("Comparison requires the reference-informed alignment study.")
    selected = [
        e for e in original["protocol"]["entries"]
        if e["identity"]["document_id"] == "3" and e["identity"]["condition"] == "G3"
        and e["packet"]["task"] == "assessment_quality"
    ]
    if len(selected) != 1:
        raise ValueError("Expected one original document-3 G3 assessment.")
    source_entry = selected[0]
    informed = [
        results[(source_entry["packet"]["packet_id"], repetition)]["judgment"]
        for repetition in range(1, 4)
    ]
    snapshots = [("original", source_entry["packet"], {
        "source_study_id": original["study_id"], "source_entry": source_entry,
        "reference_informed_judgments": informed,
    })]
    for version in ("2.1.0", "2.2.0"):
        directory, protocol = prepare_generation(
            root, source_directory, g3_supplemental_version=version,
        )
        request = GenerationRequest.model_validate(protocol["request"])
        for repetition in range(1, 4):
            record = json.loads((directory / f"generation-{repetition}.json").read_text())
            if any(record[key] != value for key, value in {
                "protocol_sha256": directory.name, "repetition": repetition, "condition": "G3",
                "model_name": protocol["original_model"],
                "rendered_prompt_sha256": protocol["rendered_prompt_sha256"],
                "prompt_components": protocol["prompt_components"],
            }.items()):
                raise ValueError("Generation provenance mismatch.")
            output = DocumentGenerationOutput.model_validate(record["output"])
            validate_output(request, output)
            packet = assessment_packet(
                protocol["source_entry"]["packet"], output,
                str(uuid5(NAMESPACE_URL, f"{directory.name}:G3:{repetition}")),
            )
            judged = json.loads((directory / f"alignment-{repetition}.json").read_text())
            if any(judged.get(key) != value for key, value in {
                "protocol_sha256": directory.name, "generation_repetition": repetition,
                "source_generation_sha256": content_hash(record),
                "input_packet_sha256": judge_input_sha256(packet),
                "prompt_artifact_sha256": protocol["judge_prompt_sha256"],
                "prompt_version": "0.7.0", "judge_model": protocol["judge_config"]["model_name"],
            }.items()):
                raise ValueError("Reference-informed judgment provenance mismatch.")
            parsed = AssessmentAlignmentJudgeOutput.model_validate(judged["result"])
            validate_alignment_evidence(parsed, packet)
            snapshots.append((f"G3-{version}-generation-{repetition}", packet, {
                "generation_protocol": protocol, "generation": record,
                "reference_informed_judgments": [judged],
            }))
    entries, sources = [], {}
    for index, (label, packet, snapshot) in enumerate(snapshots, 1):
        text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        sources[digest] = text
        blind = build_judge_input(packet, assessment_prompt_version="0.8.0")
        packet = {
            **blind, "task": "assessment_quality",
            "packet_id": str(uuid5(NAMESPACE_URL, f"blind-G3-document-3:{label}:{digest}")),
            "rubric_version": packet["rubric_version"],
        }
        identity = {
            **copy.deepcopy(source_entry["identity"]),
            "source_artifact_sha256": digest, "repetition": index, "output_variant": label,
        }
        entries.append({
            "packet": packet, "identity": identity, "generation_repetition": index,
            "input_sha256": content_hash(blind),
        })
    seed_protocol = {
        "schema_version": "1.0.0", "cohort": "playground",
        "assessment_prompt_version": "0.8.0", "generation_sources": sources, "entries": entries,
    }
    identifier = content_hash(seed_protocol)
    seed = root / "llm-judge" / "blind-g3-inputs" / identifier
    _private_directory(seed, root)
    manifest = {"study_id": identifier, "protocol": seed_protocol}
    if (seed / "manifest.json").exists():
        if json.loads((seed / "manifest.json").read_text()) != manifest:
            raise ValueError("Blind input manifest changed.")
    else:
        _write_once(seed / "manifest.json", manifest)
    return freeze_judge_study(
        root, cohort="playground", conditions=("G3",), tasks=("assessment_quality",),
        judge_repetitions=3, import_existing=False, assessment_prompt_version="0.8.0",
        source_study=seed, document_ids=("3",),
    )


def comparison(directory: Path) -> list[dict]:
    study, results = study_results(directory)
    rows = []
    for entry in study["protocol"]["entries"]:
        snapshot = json.loads(study["protocol"]["generation_sources"][
            entry["identity"]["source_artifact_sha256"]
        ])
        informed = snapshot["reference_informed_judgments"]
        blind = [
            results[(entry["packet"]["packet_id"], rep)]["judgment"]
            for rep in range(1, 4) if (entry["packet"]["packet_id"], rep) in results
        ]
        for dimension in entry["packet"]["output"]["dimensions"]:
            def verdicts(records):
                return [
                    next(d["verdict"] for d in r["result"]["dimensions"]
                         if d["dimension_id"] == dimension["dimension_id"])
                    for r in records
                ]
            rows.append({
                "output": entry["identity"]["output_variant"], "dimension": dimension["dimension_id"],
                "score": dimension["score"], "reference_informed": verdicts(informed),
                "reference_blind": verdicts(blind),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--approve-external-processing", action="store_true")
    parser.add_argument("--approve-full-batch", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        raise RuntimeError("Set PREBI_DATA_ROOT to approved protected storage.")
    directory = prepare(Path(os.environ["PREBI_DATA_ROOT"]), args.source_study)
    print("Frozen reference-blind study:", directory, flush=True)
    print(study_progress(directory), flush=True)
    if args.approve_external_processing or args.approve_full_batch:
        print(run_judge_study(
            directory, approve_external_processing=args.approve_external_processing,
            approve_full_batch=args.approve_full_batch,
        ), flush=True)
    else:
        print("Local-only. Both approval flags are required for 21 judge calls.", flush=True)
    print(json.dumps(comparison(directory), indent=2))
    print(analyze_alignment_study(directory)["dimension_stability"].to_string(index=False))


if __name__ == "__main__":
    main()
