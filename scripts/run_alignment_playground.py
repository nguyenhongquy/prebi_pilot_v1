"""Freeze and optionally execute reference-aware assessment alignment studies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from reflection_assessment_feedback.judge_stability import analyze_alignment_study
from reflection_assessment_feedback.judge_study import freeze_judge_study, run_judge_study, study_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("playground", "remaining-test"), default="playground")
    parser.add_argument("--source-study", type=Path)
    parser.add_argument("--document-id", action="append", default=None)
    parser.add_argument("--approve-external-processing", action="store_true")
    parser.add_argument("--approve-full-batch", action="store_true")
    args = parser.parse_args()
    if args.cohort == "playground" and args.source_study is None:
        parser.error("The playground pilot requires --source-study.")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("PREBI_DATA_ROOT"):
        raise RuntimeError("Set PREBI_DATA_ROOT to approved protected storage.")
    directory = freeze_judge_study(
        os.environ["PREBI_DATA_ROOT"], cohort=args.cohort,
        conditions=("G1", "G2", "G3"), tasks=("assessment_quality",),
        judge_repetitions=3, import_existing=False, assessment_prompt_version="0.7.0",
        source_study=args.source_study,
        document_ids=(
            tuple(args.document_id) if args.document_id
            else ("3",) if args.cohort == "playground" else None
        ),
    )
    print("Frozen alignment study:", directory, flush=True)
    print(study_progress(directory), flush=True)
    if args.approve_external_processing or args.approve_full_batch:
        print(run_judge_study(
            directory, approve_external_processing=args.approve_external_processing,
            approve_full_batch=args.approve_full_batch,
        ))
    else:
        print("Local-only: execution requires both explicit approval flags.")
    tables = analyze_alignment_study(directory)
    print(tables["dimension_stability"].to_string(index=False))
    print(tables["verdict_distribution"].to_string(index=False))


if __name__ == "__main__":
    main()
