"""Prospective three-generation G3 diagnostic on frozen document 3 inputs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

from reflection_assessment_feedback import EXPERT_RUBRIC, make_generation_request
from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.generator import RubricGenerationRunner, validate_output
from reflection_assessment_feedback.judge_study import (
    _private_directory, _protected, _write_once, content_hash, load_judge_study,
)
from reflection_assessment_feedback.llm_judge import (
    AssessmentAlignmentJudgeOutput, judge_input_sha256, run_quality_judge, validate_alignment_evidence,
)
from reflection_assessment_feedback.models import (
    DocumentGenerationOutput, HumanAnalysisContext, ReflectionDocument,
)
from reflection_assessment_feedback.prompt import build_prompt, prompt_components_for_request
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rate_limit import reserve_request_slot


def assessment_packet(source_packet: dict, output: DocumentGenerationOutput, packet_id: str) -> dict:
    packet = copy.deepcopy(source_packet)
    packet["packet_id"] = packet_id
    packet["output"] = {"dimensions": [
        {"dimension_id": a.dimension_id, "score": a.band, "justification": a.justification,
         "evidence_segment_ids": a.evidence_segment_ids,
         "intermediate_score_explanation": a.intermediate_score_explanation}
        for a in output.assessment.component_assessments
    ]}
    return packet


def prepare(
    root: Path, source_directory: Path, *, g3_supplemental_version: str = "2.1.0",
) -> tuple[Path, dict]:
    if g3_supplemental_version not in {"2.1.0", "2.2.0"}:
        raise ValueError("Unsupported diagnostic G3 supplemental version.")
    root = _protected(root)
    source = load_judge_study(source_directory)
    if source["protocol"]["cohort"] != "playground":
        raise ValueError("This diagnostic requires a frozen playground study.")
    entries = [
        e for e in source["protocol"]["entries"]
        if e["identity"]["document_id"] == "3" and e["identity"]["condition"] == "G3"
        and e["packet"]["task"] == "assessment_quality"
    ]
    if len(entries) != 1:
        raise ValueError("Expected exactly one document 3 G3 assessment packet.")
    entry = entries[0]
    artifact = json.loads(source["protocol"]["generation_sources"][entry["identity"]["source_artifact_sha256"]])
    original = next(p["G3"] for p in artifact["pairs"] if p["repetition"] == entry["generation_repetition"])
    request = make_generation_request(
        condition="G3",
        document=ReflectionDocument(document_id="3", segments=entry["packet"]["reflection"]["segments"]),
        rubric=EXPERT_RUBRIC,
        analysis=HumanAnalysisContext.model_validate(artifact["g3_human_analysis_context"]),
    )
    if entry["packet"]["rubric"] != request.rubric.model_dump(mode="json"):
        raise ValueError("Frozen and configured rubrics differ.")
    if hashlib.sha256(build_prompt(request).encode()).hexdigest() != original["rendered_prompt_sha256"]:
        raise ValueError("Original generation prompt cannot be reconstructed exactly.")
    config = load_experiment_config()
    if original["model_name"] != config["generation"]["model_name"]:
        raise ValueError("Configured generator differs from the original model.")
    revised = build_prompt(request, g3_supplemental_version=g3_supplemental_version)
    components = prompt_components_for_request(request, g3_supplemental_version=g3_supplemental_version)
    judge = resolve_prompt("assessment_quality_llm_judge", "0.7.0")
    protocol = {
        "schema_version": "1.0.0", "cohort": "playground", "document_id": "3", "condition": "G3",
        "generation_repetitions": 3, "judge_repetitions_per_generation": 1,
        "source_study_id": source["study_id"], "source_entry": entry,
        "source_artifact_sha256": entry["identity"]["source_artifact_sha256"],
        "request": request.model_dump(mode="json"),
        "original_output": original["output"], "original_model": original["model_name"],
        "generation_config": config["generation"], "judge_config": config["llm_judge"],
        "g3_supplemental_version": g3_supplemental_version, "rendered_prompt": revised,
        "rendered_prompt_sha256": hashlib.sha256(revised.encode()).hexdigest(),
        "prompt_components": [
            {"artifact_id": a.artifact_id, "version": a.version, "sha256": a.sha256} for a in components
        ],
        "judge_prompt_version": judge.version, "judge_prompt_sha256": judge.sha256,
        "review_policy": "Report all three generations; no preferred score or ranking; no extensions.",
    }
    identifier = content_hash(protocol)
    directory = root / "generation-diagnostics" / identifier
    _private_directory(directory, root)
    path = directory / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Frozen diagnostic protocol mismatch.")
    else:
        _write_once(path, protocol)
    return directory, protocol


def run(
    directory: Path, protocol: dict, *, approve_external_processing: bool = False,
    approve_full_batch: bool = False, generation_model=None, judge_model=None,
) -> list[dict]:
    if not approve_external_processing or not approve_full_batch:
        raise RuntimeError("Both external-processing and full-batch approval are required.")
    if content_hash(protocol) != directory.name or json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("Diagnostic protocol changed.")
    from reflection_assessment_feedback.models import GenerationRequest

    request = GenerationRequest.model_validate(protocol["request"])
    prompt = build_prompt(request, g3_supplemental_version=protocol["g3_supplemental_version"])
    if prompt != protocol["rendered_prompt"]:
        raise ValueError("Generation prompt drift.")
    if resolve_prompt("assessment_quality_llm_judge", "0.7.0").sha256 != protocol["judge_prompt_sha256"]:
        raise ValueError("Judge prompt drift.")
    if generation_model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        generation_model = ChatGoogleGenerativeAI(model=protocol["original_model"])
    if judge_model is None:
        from langchain_openai import ChatOpenAI
        judge_model = ChatOpenAI(model=protocol["judge_config"]["model_name"])
    if str(getattr(generation_model, "model", None) or getattr(generation_model, "model_name", "")) != protocol["original_model"]:
        raise ValueError("Generation model differs from the frozen model.")
    if getattr(judge_model, "model_name", "") != protocol["judge_config"]["model_name"]:
        raise ValueError("Judge model differs from the frozen model.")
    runner = RubricGenerationRunner(
        generation_model, enable_langsmith_tracing=True,
        g3_supplemental_version=protocol["g3_supplemental_version"],
    )
    root = directory.parents[1]
    summary = []
    with (directory / ".run.lock").open("a+") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Finish the prospectively fixed generations before inspecting any new judge answers.
        for repetition in range(1, 4):
            path = directory / f"generation-{repetition}.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                reserve_request_slot(root / ".rate_limits" / "gemini.state",
                                     protocol["generation_config"]["minimum_request_interval_seconds"])
                generated = runner.generate(request, repetition=repetition)
                record = {
                    **asdict(generated), "output": generated.output.model_dump(mode="json"),
                    "protocol_sha256": directory.name,
                }
                _write_once(path, record)
            if (
                record["protocol_sha256"] != directory.name or record["repetition"] != repetition
                or record["condition"] != "G3" or record["model_name"] != protocol["original_model"]
                or record["rendered_prompt_sha256"] != protocol["rendered_prompt_sha256"]
                or record["prompt_components"] != protocol["prompt_components"]
            ):
                raise ValueError("Saved generation provenance mismatch.")
            validate_output(request, DocumentGenerationOutput.model_validate(record["output"]))
        for repetition in range(1, 4):
            record = json.loads((directory / f"generation-{repetition}.json").read_text())
            output = DocumentGenerationOutput.model_validate(record["output"])
            packet = assessment_packet(
                protocol["source_entry"]["packet"], output,
                str(uuid5(NAMESPACE_URL, f"{directory.name}:G3:{repetition}")),
            )
            path = directory / f"alignment-{repetition}.json"
            if path.exists():
                judged = json.loads(path.read_text())
            else:
                reserve_request_slot(root / ".rate_limits" / "openai-judge.state",
                                     protocol["judge_config"]["minimum_request_interval_seconds"])
                judged = run_quality_judge(
                    model=judge_model, packet=packet, provider=protocol["judge_config"]["provider"],
                    model_name=protocol["judge_config"]["model_name"],
                    enable_langsmith_tracing=protocol["judge_config"]["enable_langsmith_tracing"],
                    assessment_prompt_version="0.7.0",
                )
                judged.update({
                    "protocol_sha256": directory.name, "generation_repetition": repetition,
                    "source_generation_sha256": content_hash(record),
                })
                _write_once(path, judged)
            if (
                judged["protocol_sha256"] != directory.name
                or judged["generation_repetition"] != repetition
                or judged["source_generation_sha256"] != content_hash(record)
                or judged["input_packet_sha256"] != judge_input_sha256(packet)
                or judged["prompt_artifact_sha256"] != protocol["judge_prompt_sha256"]
                or judged["judge_model"] != protocol["judge_config"]["model_name"]
            ):
                raise ValueError("Saved alignment provenance mismatch.")
            parsed = AssessmentAlignmentJudgeOutput.model_validate(judged["result"])
            validate_alignment_evidence(parsed, packet)
            summary.extend({
                "generation_repetition": repetition, "dimension_id": d.dimension_id,
                "score": d.generated_score, "alignment": d.verdict,
            } for d in parsed.dimensions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--g3-supplemental-version", choices=("2.1.0", "2.2.0"), default="2.1.0")
    parser.add_argument("--approve-external-processing", action="store_true")
    parser.add_argument("--approve-full-batch", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        raise RuntimeError("Set PREBI_DATA_ROOT to approved protected storage.")
    directory, protocol = prepare(
        Path(os.environ["PREBI_DATA_ROOT"]), args.source_study,
        g3_supplemental_version=args.g3_supplemental_version,
    )
    print("Frozen generation diagnostic:", directory, flush=True)
    print("Plan: 3 G3 generations; 1 alignment judgment per generation (3 judge calls).", flush=True)
    if args.approve_external_processing or args.approve_full_batch:
        print(json.dumps(run(
            directory, protocol, approve_external_processing=args.approve_external_processing,
            approve_full_batch=args.approve_full_batch,
        ), indent=2))
    else:
        print("Local-only. Both approval flags are required for execution.")


if __name__ == "__main__":
    main()
