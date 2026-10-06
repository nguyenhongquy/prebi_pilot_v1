from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reflection_assessment_feedback import EXPERT_RUBRIC, IntermediateAnalysis, make_generation_request
from reflection_assessment_feedback.development_cohorts import build_gold_segment_analysis_summary
from reflection_assessment_feedback.experiment_config import experiment_config_sha256, load_experiment_config
from reflection_assessment_feedback.generator import RubricGenerationRunner
from reflection_assessment_feedback.llm_judge import judge_input_sha256, run_quality_judge
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rate_limit import reserve_request_slot
from reflection_assessment_feedback.rating_packets import build_rating_packet


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generate_validated(runner, request, repetition: int, rate_state: Path, interval: float):
    for attempt in range(1, 4):
        reserve_request_slot(rate_state, interval)
        try:
            return runner.generate(request, repetition=repetition)
        except (OutputParserException, ValidationError, ValueError):
            if attempt == 3:
                raise
            print(f"Retrying invalid generated output, attempt {attempt + 1}", flush=True)


def save_private(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
        json.dump(value, temporary, indent=2, ensure_ascii=False)
    try:
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def prepare() -> tuple[dict, dict, Path, dict]:
    config = load_experiment_config()
    protected = Path(os.environ["PREBI_DATA_ROOT"]).expanduser().resolve()
    if protected == PROJECT_ROOT or PROJECT_ROOT in protected.parents:
        raise ValueError("Protected storage must be outside the source tree")
    checkpoint_directory = PROJECT_ROOT / config["paths"]["classifier_checkpoints"]
    manifest = json.loads((checkpoint_directory / "run-manifest.json").read_text())
    if manifest.get("canonical_authors") != 62 or not manifest.get("test_evaluation_completed"):
        raise ValueError("A completed author-disjoint training run is required")
    checkpoint_hashes = {}
    for name in ("scope", "direct-situationserfassung", "direct-analyse", "direct-konsequenzen"):
        digest = hashlib.sha256((checkpoint_directory / f"{name}.pt").read_bytes()).hexdigest()
        if digest != manifest["models"][name]["checkpoint_sha256"]:
            raise ValueError("Checkpoint fingerprint mismatch")
        checkpoint_hashes[name] = digest
    split_path = PROJECT_ROOT / config["paths"]["provisional_split_directory"] / "provisional_test_modeling_wide.csv"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != manifest["split_sha256"]["test_modeling_wide"]:
        raise ValueError("Checkpoint and test split do not match")
    for name, digest in manifest["split_sha256"].items():
        split, view = name.split("_", 1)
        path = split_path.parent / f"provisional_{split}_{view}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Training split fingerprint mismatch")
    notebook = json.loads((PROJECT_ROOT / "02_segment_inference.ipynb").read_text())
    namespace: dict[str, Any] = {"__name__": "__g2_direct_evaluation__"}
    for index in (1, 3, 5):
        exec(compile("".join(notebook["cells"][index]["source"]), f"inference:cell-{index}", "exec"), namespace)
    prompts = {
        task: resolve_prompt(config["prompts"][f"{task}_llm_id"], config["prompts"][f"{task}_llm_version"]).sha256
        for task in ("assessment_quality", "feedback_quality")
    }
    provenance = {
        "condition": "G2", "classifier_backend": "gbert", "classifier_architecture": "direct",
        "training_run_id": checkpoint_directory.name, "checkpoint_sha256": checkpoint_hashes,
        "test_split_sha256": split_hash, "canonical_authors": 62,
        "experiment_config_sha256": experiment_config_sha256(),
        "document_ids": sorted(namespace["DEVELOPMENT_DOCUMENT_IDS"]),
        "cohort_rule": {
            "excluded_author_document_id": str(config["dataset"]["exploratory_document_id"]),
            "excluded_document_ids": config["dataset"]["generation_excluded_document_ids"],
        },
        "teacher_reference_sha256": fingerprint(namespace["COHORT_RECORDS"][["document_id", "feedback_text"]].to_dict(orient="records")),
        "judge_prompt_sha256": prompts,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    batch = protected / "g2/author-disjoint-evaluation" / fingerprint(provenance)
    batch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (batch / "provenance.json").exists():
        save_private(batch / "provenance.json", provenance)
    print("Protected G2 evaluation batch:", batch, flush=True)
    missing = [document_id for document_id in provenance["document_ids"] if not (batch / "predictions" / f"document-{document_id}.json").exists()]
    if missing:
        for index in (7, 9, 11):
            exec(compile("".join(notebook["cells"][index]["source"]), f"inference:cell-{index}", "exec"), namespace)
        for document_id in missing:
            document = namespace["DEVELOPMENT_DOCUMENTS"][document_id]
            save_private(batch / "predictions" / f"document-{document_id}.json", {
                **provenance, "document_id": document_id,
                "document_sha256": fingerprint(document.model_dump(mode="json")),
                "analysis": namespace["PREDICTED_ANALYSES"][document_id].model_dump(mode="json"),
            })
    for document_id, document in namespace["DEVELOPMENT_DOCUMENTS"].items():
        package = json.loads((batch / "predictions" / f"document-{document_id}.json").read_text())
        if package["document_sha256"] != fingerprint(document.model_dump(mode="json")):
            raise ValueError("Cached prediction refers to different document content")
        if package["checkpoint_sha256"] != checkpoint_hashes:
            raise ValueError("Cached prediction refers to different checkpoints")
        make_generation_request(condition="G2", document=document, rubric=EXPERT_RUBRIC,
                                analysis=IntermediateAnalysis.model_validate(package["analysis"]))
    return namespace, config, batch, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "generate", "judge", "all"), default="prepare")
    parser.add_argument("--approve-external-processing", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    namespace, config, batch, provenance = prepare()
    if args.phase == "prepare":
        print("Predictions validated; no generation or judge provider calls made.", flush=True)
        return
    if not args.approve_external_processing:
        raise ValueError("Provider processing requires explicit approval")
    generation = config["generation"]
    judge = config["llm_judge"]
    if generation["provider"] != "google" or judge["provider"] != "openai":
        raise ValueError("This runner requires configured Google generation and OpenAI judging")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Approved generation and judge credentials are required")
    protected = batch.parents[2]
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    generator = RubricGenerationRunner(
        ChatGoogleGenerativeAI(model=generation["model_name"], temperature=generation["temperature"]),
        enable_langsmith_tracing=generation["enable_langsmith_tracing"],
    )
    judge_model = ChatOpenAI(model=judge["model_name"])
    ratings = []
    expected = len(provenance["document_ids"]) * generation["repetitions"] * 2
    for number, document_id in enumerate(provenance["document_ids"], start=1):
        document = namespace["DEVELOPMENT_DOCUMENTS"][document_id]
        prediction = json.loads((batch / "predictions" / f"document-{document_id}.json").read_text())
        analysis = IntermediateAnalysis.model_validate(prediction["analysis"])
        for repetition in range(1, generation["repetitions"] + 1):
            generated_path = batch / "generation" / f"document-{document_id}-rep-{repetition}.json"
            if not generated_path.exists():
                if args.phase == "judge":
                    raise FileNotFoundError("Generate all G2 feedback before judging")
                run = generate_validated(
                    generator, make_generation_request(condition="G2", document=document, rubric=EXPERT_RUBRIC, analysis=analysis),
                    repetition, protected / ".rate_limits/gemini.state", generation["minimum_request_interval_seconds"],
                )
                run_record = asdict(run)
                run_record["output"] = run.output.model_dump(mode="json")
                save_private(generated_path, {
                    "provenance": provenance, "document_id": document_id,
                    "prediction_sha256": fingerprint(prediction), "run": run_record,
                })
                print(f"Generated G2 {number}/{len(provenance['document_ids'])}, repetition {repetition}", flush=True)
            envelope = json.loads(generated_path.read_text())
            if envelope["prediction_sha256"] != fingerprint(prediction) or envelope["provenance"] != provenance:
                raise ValueError("Cached generation has mismatched prediction provenance")
            if args.phase == "generate":
                continue
            reference = namespace["COHORT_RECORDS"].loc[namespace["COHORT_RECORDS"].document_id.eq(document_id), "feedback_text"].iloc[0]
            for task in ("assessment_quality", "feedback_quality"):
                packet_path = batch / "packets" / f"document-{document_id}-rep-{repetition}-{task}.json"
                if not packet_path.exists():
                    packet = build_rating_packet(task=task, run=envelope["run"], reflection_segments=[segment.model_dump(mode="json") for segment in document.segments], rubric=EXPERT_RUBRIC.model_dump(mode="json"))
                    packet["reference_teacher_feedback"] = reference
                    if task == "assessment_quality":
                        packet["gold_analysis_summary"] = build_gold_segment_analysis_summary(namespace["SPLIT_PATH"], document_id)
                    save_private(packet_path, packet)
                packet = json.loads(packet_path.read_text())
                result_path = batch / "judge" / packet_path.name
                if not result_path.exists():
                    for attempt in range(1, 4):
                        reserve_request_slot(protected / ".rate_limits/openai-judge.state", judge["minimum_request_interval_seconds"])
                        try:
                            result = run_quality_judge(model=judge_model, packet=packet, provider=judge["provider"], model_name=judge["model_name"], enable_langsmith_tracing=judge["enable_langsmith_tracing"])
                            break
                        except (OutputParserException, ValidationError, ValueError):
                            if attempt == 3:
                                raise
                            print(f"Retrying invalid structured rating: {task}, attempt {attempt + 1}", flush=True)
                    result.update(evaluation_document_id=document_id, evaluation_condition="G2", evaluation_mode="test", repetition=repetition, source_generation_sha256=fingerprint(envelope), experiment_config_sha256=provenance["experiment_config_sha256"], training_run_id=provenance["training_run_id"])
                    save_private(result_path, result)
                    print(f"Judged G2 {number}/{len(provenance['document_ids'])}: {task}", flush=True)
                result = json.loads(result_path.read_text())
                if result["input_packet_sha256"] != judge_input_sha256(packet) or result["source_generation_sha256"] != fingerprint(envelope):
                    raise ValueError("Cached judge result has mismatched inputs")
                ratings.append(result)
    if args.phase == "generate":
        print("G2 generation completed; judge calls not requested.", flush=True)
        return
    assert len(ratings) == expected
    scores = defaultdict(list)
    unable = defaultdict(int)
    for result in ratings:
        for rating in result["result"]["ratings"]:
            key = (result["task"], rating["criterion_id"])
            if rating["unable_to_judge"]:
                unable[key] += 1
            else:
                scores[key].append(rating["score"])
    rows = []
    for task, criterion in sorted(set(scores) | set(unable)):
        values = scores[(task, criterion)]
        rows.append({"task": task, "criterion": criterion, "ratings": len(values), "unable_to_judge": unable[(task, criterion)], "mean_score": sum(values) / len(values) if values else None})
    summary = {"status": "completed", "condition": "G2", "documents": len(provenance["document_ids"]), "judge_records": len(ratings), "provenance": provenance, "protected_batch": str(batch), "criteria": rows, "completed_at": datetime.now(timezone.utc).isoformat()}
    report_directory = PROJECT_ROOT / "artifacts/g2-author-disjoint-judge" / batch.name
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(report_directory / "criteria-summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("Completed G2 judge evaluation:", report_directory, flush=True)


if __name__ == "__main__":
    main()