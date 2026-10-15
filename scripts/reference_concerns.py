"""Extract reference-only draft inventories or explicitly approve reviewed drafts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.judge_artifacts import content_hash
from reflection_assessment_feedback.judge_study import load_judge_study
from reflection_assessment_feedback.reference_inventory import (
    ExtractorConfig, approve_inventory, extract_reference_inventory,
    reference_input_from_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    extract = actions.add_parser("extract")
    extract.add_argument("--source-study", type=Path, required=True)
    extract.add_argument("--document-id", action="append", required=True)
    extract.add_argument("--extractor-prompt-version", default="0.1.0")
    extract.add_argument("--extraction-revision", default="1")
    extract.add_argument("--approve-external-processing", action="store_true")
    approve = actions.add_parser("approve")
    approve.add_argument("--inventory-id", required=True)
    approve.add_argument("--approved-by", required=True)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        raise RuntimeError("Set PREBI_DATA_ROOT to approved protected storage.")
    root = Path(os.environ["PREBI_DATA_ROOT"])
    if args.action == "approve":
        print(approve_inventory(
            root / "llm-judge" / "reference-inventories" / args.inventory_id,
            approved_by=args.approved_by,
        ))
        return
    protocol = load_judge_study(args.source_study)["protocol"]
    sources = {}
    for entry in protocol["entries"]:
        document = entry["identity"]["document_id"]
        if document not in args.document_id:
            continue
        source = reference_input_from_packet(entry["packet"], document)
        if document in sources and content_hash(sources[document]) != content_hash(source):
            raise ValueError("Same reflection has conflicting reference-only source inputs.")
        sources[document] = source
    if set(sources) != set(args.document_id):
        raise ValueError("Requested reflections are missing from source study.")
    judge_config = load_experiment_config()["llm_judge"]
    config = ExtractorConfig(
        provider=judge_config["provider"], model_name=judge_config["model_name"],
        prompt_version=args.extractor_prompt_version,
        extraction_revision=args.extraction_revision,
    )
    for document, source in sources.items():
        directory = extract_reference_inventory(
            root, source=source, config=config,
            approve_external_processing=args.approve_external_processing,
            enable_langsmith_tracing=judge_config["enable_langsmith_tracing"],
            minimum_request_interval_seconds=judge_config["minimum_request_interval_seconds"],
        )
        print(f"Document {document}: draft inventory {directory}", flush=True)
    print("No approvals or feedback judgments were performed.", flush=True)


if __name__ == "__main__":
    main()
