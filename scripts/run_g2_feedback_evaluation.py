"""Prepare current G2 generation, reference inventories, and feedback-only judging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

from reflection_assessment_feedback.experiment_config import PROJECT_ROOT, load_experiment_config
from reflection_assessment_feedback.generator import validate_output
from reflection_assessment_feedback.judge_artifacts import (
    content_hash, private_directory, protected_path, write_once,
)
from reflection_assessment_feedback.judge_study import (
    freeze_judge_study, run_judge_study, study_progress,
)
from reflection_assessment_feedback.llm_judge import build_judge_input
from reflection_assessment_feedback.models import DocumentGenerationOutput, IntermediateAnalysis
from reflection_assessment_feedback import make_generation_request
from reflection_assessment_feedback.pairwise_packets import _validate_run_provenance
from reflection_assessment_feedback.rating_packets import build_rating_packet
from reflection_assessment_feedback.reference_inventory import (
    ExtractorConfig, extract_reference_inventory, inventory_reference,
    inventory_specification, load_inventory_artifact, reference_input_from_packet,
)
from reflection_assessment_feedback.rubric_registry import resolve_rubric
from scripts.evaluate_g2_direct_gbert import fingerprint, prepare


def prepare_sources(namespace: dict, config: dict) -> dict[str, dict]:
    rubric = resolve_rubric(
        config["generation"]["rubric_id"], config["generation"]["expected_rubric_version"],
    ).rubric.model_dump(mode="json")
    sources = {}
    for document_id in namespace["DEVELOPMENT_DOCUMENT_IDS"]:
        document = namespace["DEVELOPMENT_DOCUMENTS"][document_id]
        reference = namespace["COHORT_RECORDS"].loc[
            namespace["COHORT_RECORDS"].document_id.eq(document_id), "feedback_text",
        ].iloc[0]
        sources[document_id] = reference_input_from_packet({
            "reflection": {"segments": [segment.model_dump(mode="json") for segment in document.segments]},
            "reference_teacher_feedback": reference,
            "rubric": rubric,
            "rubric_version": rubric["version"],
        }, document_id)
    return sources


def select_inventories(root: Path, sources: dict[str, dict], config: dict) -> tuple[dict, dict]:
    judge = config["llm_judge"]
    extractor = ExtractorConfig(
        provider=judge["provider"], model_name=judge["model_name"], prompt_version="0.2.0",
    )
    references, pending = {}, {}
    for document_id, source in sources.items():
        identifier = content_hash(inventory_specification(source, extractor))
        directory = root / "llm-judge" / "reference-inventories" / identifier
        if not (directory / "draft.json").exists():
            pending[document_id] = {"inventory_id": identifier, "status": "missing"}
            continue
        record = load_inventory_artifact(directory)
        if not (directory / "approval.json").exists():
            pending[document_id] = {"inventory_id": identifier, "status": "needs_review"}
            continue
        load_inventory_artifact(directory, require_approved=True)
        references[document_id] = inventory_reference(record)
    return references, pending


def freeze_g2_feedback(
    root: Path, namespace: dict, config: dict, batch: Path,
    provenance: dict, references: dict,
) -> Path:
    if set(references) != set(provenance["document_ids"]):
        raise ValueError("Review and approve one matching inventory per reflection before judging.")
    rubric = resolve_rubric(
        config["generation"]["rubric_id"], config["generation"]["expected_rubric_version"],
    )
    entries, sources = [], {}
    for document_id in provenance["document_ids"]:
        prediction_path = batch / "predictions" / f"document-{document_id}.json"
        prediction = json.loads(prediction_path.read_text())
        for repetition in range(1, config["generation"]["repetitions"] + 1):
            generation_path = batch / "generation" / f"document-{document_id}-rep-{repetition}.json"
            text = generation_path.read_text(encoding="utf-8")
            envelope = json.loads(text)
            if (
                envelope["provenance"] != provenance
                or envelope["prediction_sha256"] != fingerprint(prediction)
                or envelope["document_id"] != document_id
            ):
                raise ValueError("G2 generation source differs from its frozen prediction or batch.")
            run = envelope["run"]
            _validate_run_provenance(
                run, condition="G2", config=config, rubric_sha256=rubric.sha256,
            )
            document = namespace["DEVELOPMENT_DOCUMENTS"][document_id]
            request = make_generation_request(
                condition="G2", document=document, rubric=rubric.rubric,
                analysis=IntermediateAnalysis.model_validate(prediction["analysis"]),
            )
            validate_output(request, DocumentGenerationOutput.model_validate(run["output"]))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            sources[digest] = text
            packet = build_rating_packet(
                task="feedback_quality", run=run,
                reflection_segments=[segment.model_dump(mode="json") for segment in document.segments],
                rubric=rubric.rubric.model_dump(mode="json"),
            )
            packet["packet_id"] = str(uuid5(NAMESPACE_URL, f"G2-feedback:{digest}:{document_id}:{repetition}"))
            packet["reference_teacher_feedback"] = namespace["COHORT_RECORDS"].loc[
                namespace["COHORT_RECORDS"].document_id.eq(document_id), "feedback_text",
            ].iloc[0]
            entries.append({
                "packet": packet,
                "identity": {
                    "document_id": document_id, "condition": "G2", "repetition": repetition,
                    "cohort_id": config["dataset"]["generation_cohort"],
                    "run_id": run["run_id"], "source_artifact_sha256": digest,
                    "task": "feedback_quality",
                },
                "input_sha256": content_hash(build_judge_input(packet)),
            })
    seed_protocol = {
        "schema_version": "1.0.0", "cohort": "remaining-test",
        "entries": entries, "generation_sources": sources,
    }
    identifier = content_hash(seed_protocol)
    directory = root / "llm-judge" / "g2-feedback-inputs" / identifier
    private_directory(directory, root)
    manifest = {"study_id": identifier, "protocol": seed_protocol}
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen G2 feedback inputs changed.")
    else:
        write_once(path, manifest)
    return freeze_judge_study(
        root, cohort="remaining-test", conditions=("G2",), tasks=("feedback_quality",),
        judge_repetitions=1, import_existing=False, feedback_prompt_version="0.12.0",
        source_study=directory, document_ids=tuple(provenance["document_ids"]),
        reference_inventory_refs=references,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "generate", "inventories", "freeze", "judge"), default="prepare")
    parser.add_argument("--approve-external-processing", action="store_true")
    parser.add_argument("--approve-full-batch", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        parser.error("Set PREBI_DATA_ROOT to approved protected storage.")
    root = protected_path(os.environ["PREBI_DATA_ROOT"])
    config = load_experiment_config()
    if config["generation"]["expected_prompt_version"] != "2.0.0":
        parser.error("This workflow requires generation prompt 2.0.0.")
    external = args.phase in {"generate", "inventories", "judge"}
    if external and not (args.approve_external_processing and args.approve_full_batch):
        parser.error("External phases require both processing and full-batch approval flags.")
    namespace, config, batch, provenance = prepare("remaining-test")
    sources = prepare_sources(namespace, config)
    references, pending = select_inventories(root, sources, config)
    print(f"G2 batch: {batch}", flush=True)
    print(f"Approved matching inventories: {len(references)}; pending: {len(pending)}", flush=True)
    if args.phase == "prepare":
        missing = sum(
            not (batch / "generation" / f"document-{document}-rep-{repetition}.json").exists()
            for document in provenance["document_ids"]
            for repetition in range(1, config["generation"]["repetitions"] + 1)
        )
        print(f"Uncached G2 generations: {missing}; maximum G2 feedback judgments: {len(sources)}")
        print(json.dumps(pending, indent=2))
        print("Local preparation complete; no provider calls made.", flush=True)
    elif args.phase == "generate":
        subprocess.run([
            sys.executable, "-m", "scripts.evaluate_g2_direct_gbert",
            "--phase", "generate", "--cohort", "remaining-test", "--approve-external-processing",
        ], check=True, cwd=PROJECT_ROOT)
    elif args.phase == "inventories":
        judge = config["llm_judge"]
        extractor = ExtractorConfig(
            provider=judge["provider"], model_name=judge["model_name"], prompt_version="0.2.0",
        )
        for document_id, item in pending.items():
            if item["status"] == "missing":
                directory = extract_reference_inventory(
                    root, source=sources[document_id], config=extractor,
                    approve_external_processing=True,
                    enable_langsmith_tracing=judge["enable_langsmith_tracing"],
                    minimum_request_interval_seconds=judge["minimum_request_interval_seconds"],
                )
                print(f"Document {document_id}: draft {directory.name}", flush=True)
            else:
                print(f"Document {document_id}: existing draft {item['inventory_id']} needs review", flush=True)
        print("No inventories automatically approved; review drafts before freezing.", flush=True)
    else:
        directory = freeze_g2_feedback(root, namespace, config, batch, provenance, references)
        print(f"Frozen G2-only feedback study: {directory}", flush=True)
        print(study_progress(directory), flush=True)
        if args.phase == "judge":
            print(run_judge_study(
                directory, approve_external_processing=True, approve_full_batch=True,
            ), flush=True)


if __name__ == "__main__":
    main()
