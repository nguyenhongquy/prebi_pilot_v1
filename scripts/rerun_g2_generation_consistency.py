from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reflection_assessment_feedback import EXPERT_RUBRIC, IntermediateAnalysis, make_generation_request
from reflection_assessment_feedback.data import load_human_analysis_context
from reflection_assessment_feedback.development_cohorts import load_remaining_test_feedback_cohort
from reflection_assessment_feedback.experiment_config import load_experiment_config, experiment_config_sha256
from reflection_assessment_feedback.generation_consistency import consistency_tables, paired_consistency_differences
from reflection_assessment_feedback.generator import RubricGenerationRunner, validate_output
from reflection_assessment_feedback.judge_analysis import load_new_g2_judge_scores
from reflection_assessment_feedback.models import GenerationOutput, GenerationRequest
from reflection_assessment_feedback.prompt import build_prompt, prompt_components_for_request
from scripts.evaluate_g2_direct_gbert import generate_validated, save_private

HISTORICAL_PROTOCOL = "17206b4b3a6d4d5d77d32f58e9a14b6db5e3119c6c8bf5bd5ce5a785ab451321"
NEW_G2_BATCH = "c9c3a10dfdd8756be0206e93e5025af7d3cd2629ed3a34397db88f6ebfdabf6a"
NEW_VARIANT = "G2 new direct GBERT"
VARIANTS = ("G1 historical", "G2 historical", "G3 historical", NEW_VARIANT)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def repetition_key(document_id: str, condition: str, repetition: int) -> str:
    return f"{hashlib.sha256(document_id.encode()).hexdigest()[:20]}-{condition}-{repetition}"


def append_scores(rows: list, output: GenerationOutput, document_id: str, condition: str, repetition: int) -> None:
    for assessment in output.assessment.component_assessments:
        rows.append({"document_id": document_id, "condition": condition,
                     "dimension_id": assessment.dimension_id, "repetition": repetition, "score": float(assessment.band)})


def prepare() -> dict:
    config = load_experiment_config()
    protected = Path(os.environ["PREBI_DATA_ROOT"]).expanduser().resolve()
    if protected == PROJECT_ROOT or PROJECT_ROOT in protected.parents:
        raise ValueError("Protected storage must be outside the source tree")
    historical_directory = protected / "generation-consistency" / HISTORICAL_PROTOCOL
    historical_path = historical_directory / "protocol.json"
    historical = json.loads(historical_path.read_text())
    if canonical_hash(historical) != HISTORICAL_PROTOCOL:
        raise ValueError("Historical frozen protocol hash mismatch")
    if historical["repetitions"] != 5 or historical["category_policy"] != "exact_decimal" or historical["band_mapping"] is not None:
        raise ValueError("Expected the historical five-repeat exact-decimal protocol")
    if historical["generation_config"] != config["generation"]:
        raise ValueError("Historical and new generator settings differ")
    summary_path = PROJECT_ROOT / "artifacts/g2-author-disjoint-judge" / NEW_G2_BATCH / "summary.json"
    new_scores, _ = load_new_g2_judge_scores(summary_path, protected)
    summary = json.loads(summary_path.read_text())
    prediction_batch = Path(summary["protected_batch"])
    cohort = load_remaining_test_feedback_cohort()
    document_ids = set(cohort.document_id)
    historical_ids = {str(row["document_id"]) for row in historical["cohort"] if row["mode"] == "test"}
    if document_ids != historical_ids or set(new_scores.document_id) != document_ids:
        raise ValueError("New G2 and historical consistency test cohorts differ")
    split_directory = PROJECT_ROOT / config["paths"]["provisional_split_directory"]
    manifest = pd.read_csv(split_directory / "provisional_split_manifest.csv", dtype={"document_id": str, "author_id": str})
    units = manifest.loc[manifest.document_id.isin(document_ids), ["document_id", "author_id"]].copy()
    units["mode"] = "test"
    units = units.sort_values("document_id").reset_index(drop=True)
    requests = {}
    historical_requests = {}
    prediction_hashes = {}
    historical_rows = []
    for document_id in units.document_id:
        document, _ = load_human_analysis_context(split_directory / "provisional_test_modeling_wide.csv", document_id)
        prediction_path = prediction_batch / "predictions" / f"document-{document_id}.json"
        package = json.loads(prediction_path.read_text())
        request = make_generation_request(condition="G2", document=document, rubric=EXPERT_RUBRIC,
                                          analysis=IntermediateAnalysis.model_validate(package["analysis"]))
        requests[document_id] = request
        prediction_hashes[document_id] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        for condition in ("G1", "G2", "G3"):
            snapshot_path = historical_directory / "inputs" / f"{repetition_key(document_id, condition, 0)}.json"
            snapshot = json.loads(snapshot_path.read_text())
            identity = f"{document_id}:{condition}"
            if canonical_hash(snapshot) != historical["request_hashes"][identity]:
                raise ValueError("Historical frozen input hash mismatch")
            historical_request = GenerationRequest.model_validate(snapshot)
            if historical_request.document != document or historical_request.rubric != EXPERT_RUBRIC:
                raise ValueError("Historical and new reflection or rubric differs")
            components = [{"id": item.artifact_id, "version": item.version, "sha256": item.sha256}
                          for item in prompt_components_for_request(historical_request)]
            if components != historical["prompt_components"][condition]:
                raise ValueError("Historical and current prompt artifacts differ")
            historical_requests[(document_id, condition)] = historical_request
            for repetition in range(1, 6):
                path = historical_directory / "results" / f"{repetition_key(document_id, condition, repetition)}.json"
                record = json.loads(path.read_text())
                expected = {"protocol_sha256": HISTORICAL_PROTOCOL, "document_id": document_id,
                            "condition": condition, "repetition": repetition,
                            "request_sha256": historical["request_hashes"][identity],
                            "prompt_sha256": historical["prompt_hashes"][identity]}
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError("Historical repetition provenance mismatch")
                if record["run"]["model_name"] != config["generation"]["model_name"]:
                    raise ValueError("Historical generator model differs")
                output = GenerationOutput.model_validate(record["output"])
                validate_output(historical_request, output)
                append_scores(historical_rows, output, document_id, f"{condition} historical", repetition)
    protocol = {
        "experiment_tag": "g2-author-disjoint-consistency-v1", "repetitions": 5, "maximum_attempts": 3,
        "category_policy": "exact_decimal", "band_mapping": None, "variants": list(VARIANTS),
        "historical_protocol_sha256": HISTORICAL_PROTOCOL,
        "historical_protocol_file_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        "generation_config": config["generation"], "experiment_config_sha256": experiment_config_sha256(),
        "prediction_batch": str(prediction_batch), "prediction_sha256": prediction_hashes,
        "checkpoint_sha256": summary["provenance"]["checkpoint_sha256"],
        "cohort": units.to_dict(orient="records"),
        "request_hashes": {document_id: canonical_hash(request.model_dump(mode="json")) for document_id, request in requests.items()},
        "prompt_hashes": {document_id: hashlib.sha256(build_prompt(request).encode()).hexdigest() for document_id, request in requests.items()},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    protocol_hash = canonical_hash(protocol)
    directory = protected / "generation-consistency-g2-author-disjoint" / protocol_hash
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (directory / "protocol.json").exists():
        save_private(directory / "protocol.json", protocol)
    elif json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("New frozen protocol mismatch")
    for document_id, request in requests.items():
        path = directory / "inputs" / f"document-{document_id}.json"
        if not path.exists():
            save_private(path, request.model_dump(mode="json"))
        elif json.loads(path.read_text()) != request.model_dump(mode="json"):
            raise ValueError("New frozen request mismatch")
    print("Validated 270 historical test generations; new G2 needs 90 independent generations.", flush=True)
    print("Frozen new G2 consistency protocol:", directory, flush=True)
    return {"config": config, "protected": protected, "protocol": protocol, "protocol_hash": protocol_hash,
            "directory": directory, "requests": requests, "units": units, "historical_rows": historical_rows}


def validate_new_record(record: dict, context: dict, document_id: str, repetition: int) -> GenerationOutput:
    expected = {"protocol_sha256": context["protocol_hash"], "document_id": document_id,
                "condition": "G2", "repetition": repetition,
                "request_sha256": context["protocol"]["request_hashes"][document_id],
                "prompt_sha256": context["protocol"]["prompt_hashes"][document_id]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Cached new G2 repetition provenance mismatch")
    output = GenerationOutput.model_validate(record["output"])
    validate_output(context["requests"][document_id], output)
    return output


def analyze(context: dict) -> Path:
    rows = list(context["historical_rows"])
    for document_id in context["units"].document_id:
        for repetition in range(1, 6):
            path = context["directory"] / "results" / f"document-{document_id}-rep-{repetition}.json"
            record = json.loads(path.read_text())
            output = validate_new_record(record, context, document_id, repetition)
            append_scores(rows, output, document_id, NEW_VARIANT, repetition)
    scores = pd.DataFrame(rows)
    tables = consistency_tables(scores, units=context["units"], repetitions=5, conditions=VARIANTS)
    differences = paired_consistency_differences(
        tables["per_document"], samples=2000, seed=context["config"]["evaluation"]["random_seed"],
        contrasts=tuple((historical, NEW_VARIANT) for historical in VARIANTS[:-1]),
    )
    if not tables["within"]["complete_documents"].eq(18).all():
        raise ValueError("Consistency comparison requires all five repeats on every document")
    private_analysis = context["directory"] / "analysis"
    private_analysis.mkdir(exist_ok=True, mode=0o700)
    scores.to_csv(private_analysis / "scores.csv", index=False)
    tables["per_document"].to_csv(private_analysis / "per-document.csv", index=False)
    report_directory = PROJECT_ROOT / "artifacts/g2-generation-consistency" / context["protocol_hash"]
    report_directory.mkdir(parents=True, exist_ok=True)
    for name in ("within", "between", "frequencies"):
        tables[name].to_csv(report_directory / f"{name}.csv", index=False)
    differences.to_csv(report_directory / "paired-differences.csv", index=False)
    summary = {"status": "completed", "documents": 18, "repetitions": 5,
               "new_g2_generations": 90, "historical_generations": 270, "category_policy": "exact_decimal",
               "protocol_sha256": context["protocol_hash"], "protected_directory": str(context["directory"]),
               "historical_protocol_sha256": HISTORICAL_PROTOCOL, "author_clusters": context["units"].author_id.nunique(),
               "within": tables["within"].to_dict(orient="records")}
    (report_directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(tables["within"].to_string(index=False), flush=True)
    print("Completed consistency comparison:", report_directory, flush=True)
    return report_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "generate", "analyze"), default="prepare")
    parser.add_argument("--approve-external-processing", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    context = prepare()
    if args.phase == "prepare":
        print("Local preflight complete; no provider calls made.", flush=True)
        return
    if args.phase == "generate":
        if not args.approve_external_processing:
            raise ValueError("External generation requires explicit approval")
        from langchain_google_genai import ChatGoogleGenerativeAI
        generation = context["config"]["generation"]
        runner = RubricGenerationRunner(
            ChatGoogleGenerativeAI(model=generation["model_name"], temperature=generation["temperature"]),
            enable_langsmith_tracing=generation["enable_langsmith_tracing"],
        )
        count = 0
        for repetition in range(1, 6):
            document_ids = context["units"].document_id.tolist()
            shift = (repetition - 1) % len(document_ids)
            for document_id in document_ids[shift:] + document_ids[:shift]:
                path = context["directory"] / "results" / f"document-{document_id}-rep-{repetition}.json"
                if path.exists():
                    validate_new_record(json.loads(path.read_text()), context, document_id, repetition)
                    count += 1
                    continue
                run = generate_validated(runner, context["requests"][document_id], repetition,
                                         context["protected"] / ".rate_limits/gemini.state", generation["minimum_request_interval_seconds"])
                serialized = asdict(run)
                serialized["output"] = run.output.model_dump(mode="json")
                record = {"protocol_sha256": context["protocol_hash"], "document_id": document_id,
                          "condition": "G2", "repetition": repetition,
                          "request_sha256": context["protocol"]["request_hashes"][document_id],
                          "prompt_sha256": context["protocol"]["prompt_hashes"][document_id],
                          "run": serialized, "output": serialized["output"]}
                validate_new_record(record, context, document_id, repetition)
                save_private(path, record)
                count += 1
                print(f"New G2 consistency: {count}/90 saved; repetition {repetition}", flush=True)
    analyze(context)


if __name__ == "__main__":
    main()