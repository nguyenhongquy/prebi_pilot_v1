"""Pilot human-reference feedback alignment on original playground G1/G2/G3 outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from reflection_assessment_feedback.judge_stability import analyze_judge_study
from reflection_assessment_feedback.judge_study import (
    freeze_judge_study, run_judge_study, study_progress,
)
from reflection_assessment_feedback.reference_inventory import load_inventory_artifact, inventory_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", default="3")
    parser.add_argument("--additional-document-id", action="append", default=[])
    parser.add_argument("--feedback-prompt-version", choices=("0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"), default="0.4.0")
    parser.add_argument("--reference-inventory-id", action="append", default=[], metavar="DOCUMENT=SHA256")
    parser.add_argument("--judge-repetitions", type=int, default=3)
    parser.add_argument("--approve-external-processing", action="store_true")
    parser.add_argument("--approve-full-batch", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        raise RuntimeError("Set PREBI_DATA_ROOT to approved protected storage.")
    references = {}
    for item in args.reference_inventory_id:
        document, identifier = item.split("=", 1)
        if document in references:
            raise ValueError("Duplicate inventory selection for a document.")
        record = load_inventory_artifact(
            Path(os.environ["PREBI_DATA_ROOT"]) / "llm-judge" / "reference-inventories" / identifier,
            require_approved=True,
        )
        references[document] = inventory_reference(record)
    directory = freeze_judge_study(
        os.environ["PREBI_DATA_ROOT"], cohort="playground", conditions=("G1", "G2", "G3"),
        tasks=("feedback_quality",), judge_repetitions=args.judge_repetitions, import_existing=False,
        feedback_prompt_version=args.feedback_prompt_version,
        document_ids=(args.document_id, *args.additional_document_id),
        reference_inventory_refs=references if args.reference_inventory_id else None,
    )
    print("Frozen feedback-alignment pilot:", directory, flush=True)
    print(study_progress(directory), flush=True)
    if args.approve_external_processing or args.approve_full_batch:
        print(run_judge_study(
            directory, approve_external_processing=args.approve_external_processing,
            approve_full_batch=args.approve_full_batch,
        ), flush=True)
    else:
        print("Local-only. Both approval flags are required to execute the planned calls.", flush=True)
    tables = analyze_judge_study(directory)
    print(json.dumps(tables["decisions"].to_dict(orient="records"), indent=2, ensure_ascii=False))
    if "stability" in tables:
        print(tables["stability"].to_string(index=False))
    else:
        print(tables["issues"].to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
