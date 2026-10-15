from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from reflection_assessment_feedback.experiment_config import experiment_config_sha256, load_experiment_config
from reflection_assessment_feedback.judge_batch import build_judge_batch
from reflection_assessment_feedback.llm_judge import (
    AssessmentAlignmentJudgeOutput, FeedbackQualityJudgeOutput, FeedbackReferenceAlignmentOutput,
    validate_feedback_alignment, _validate_span_comments,
    FeedbackIssueJudgeOutput, validate_feedback_issues,
    judge_output_model, validate_alignment_evidence,
    build_judge_input, judge_input_sha256,
    FrozenConcernJudgeOutput, validate_frozen_concern_output,
)
from reflection_assessment_feedback.prompt_registry import resolve_prompt


def expected_judge_metadata(packet: dict, identity: dict, config: dict, cohort: str) -> dict:
    task = packet["task"]
    prompt = resolve_prompt(
        config["prompts"][f"{task}_llm_id"], config["prompts"][f"{task}_llm_version"],
    )
    blind = task == "assessment_quality" and prompt.version == "0.8.0"
    canonical_gold = json.dumps(
        packet.get("gold_analysis_summary"), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    )
    metadata = {
        "record_type": "llm_judge_quality_rating",
        "packet_id": packet["packet_id"], "task": task,
        "evaluation_mode": "playground" if cohort == "playground" else "test",
        "evaluation_document_id": identity["document_id"],
        "evaluation_cohort_id": identity["cohort_id"],
        "source_artifact_sha256": identity["source_artifact_sha256"],
        "input_packet_sha256": judge_input_sha256(
            packet, assessment_prompt_version=prompt.version if task == "assessment_quality" else None,
            feedback_prompt_version=prompt.version if task == "feedback_quality" else None,
        ),
        "teacher_reference_sha256": (
            None if blind else hashlib.sha256(packet["reference_teacher_feedback"].encode()).hexdigest()
        ),
        "gold_analysis_summary_sha256": (
            hashlib.sha256(canonical_gold.encode()).hexdigest()
            if task == "assessment_quality" and not blind else None
        ),
        "judge_provider": config["llm_judge"]["provider"],
        "judge_model": config["llm_judge"]["model_name"],
        "prompt_id": prompt.artifact_id, "prompt_version": prompt.version,
        "prompt_artifact_sha256": prompt.sha256,
        "experiment_config_sha256": experiment_config_sha256(),
    }
    if task == "feedback_quality" and prompt.version in {"0.8.0", "0.9.0", "0.10.0", "0.11.0"}:
        metadata["reference_inventory_ref"] = packet["reference_inventory_ref"]
        metadata["schema_version"] = {
            "0.8.0": "2.0.0", "0.9.0": "2.1.0", "0.10.0": "2.2.0", "0.11.0": "2.3.0",
        }[prompt.version]
    return metadata


def validate_judge_record(record: dict, packet: dict, expected: dict):
    if any(record.get(field) != value for field, value in expected.items()):
        raise ValueError("Judge input or provenance mismatch.")
    output = judge_output_model(packet["task"], record["prompt_version"]).model_validate(record["result"])
    if isinstance(output, FeedbackQualityJudgeOutput):
        _validate_span_comments(output, build_judge_input(packet))
    if isinstance(output, AssessmentAlignmentJudgeOutput):
        validate_alignment_evidence(output, packet)
    if isinstance(output, FeedbackReferenceAlignmentOutput):
        validate_feedback_alignment(output, packet)
    if isinstance(output, FeedbackIssueJudgeOutput):
        validate_feedback_issues(output, packet)
    if isinstance(output, FrozenConcernJudgeOutput):
        validate_frozen_concern_output(output, packet)
    return output


def load_selected_judge_results(
    protected_root: str | Path,
    *,
    cohort: Literal["playground", "remaining-test"] = "playground",
    conditions: tuple[str, ...] = ("G1", "G2", "G3"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join current packets to saved judgments across batches without provider calls."""
    packets, key = build_judge_batch(
        protected_root, cohort=cohort, conditions=conditions, pilot_only=False,
    )
    root = Path(protected_root).expanduser().resolve()
    config = load_experiment_config()
    matches: dict[str, list[tuple[Path, dict]]] = {p["packet_id"]: [] for p in packets}
    for path in sorted((root / "llm-judge" / "development").glob("*/judge-*.json")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Judge result resolves outside protected storage.")
        record = json.loads(path.read_text())
        if record.get("packet_id") in matches:
            matches[record["packet_id"]].append((path, record))
    rows, coverage, duplicates = [], [], []
    for packet in packets:
        identifier = packet["packet_id"]
        identity = key[identifier]
        task = packet["task"]
        expected = expected_judge_metadata(packet, identity, config, cohort)
        validated = []
        for path, record in matches[identifier]:
            validated.append(validate_judge_record(record, packet, expected))
        base = {
            "cohort": cohort, "document_id": identity["document_id"],
            "condition": identity["condition"], "repetition": identity["repetition"],
            "task": task, "packet_id": identifier,
        }
        count = len(validated)
        coverage.append({**base, "saved_judgments": count,
                         "status": "missing" if count == 0 else "complete" if count == 1 else "duplicate"})
        if count > 1:
            duplicates.extend({**base, "result_path": str(path)} for path, _ in matches[identifier])
        if count == 1:
            if isinstance(validated[0], (AssessmentAlignmentJudgeOutput, FeedbackReferenceAlignmentOutput, FeedbackIssueJudgeOutput, FrozenConcernJudgeOutput)):
                raise ValueError("Use the frozen alignment-study analysis for alignment records.")
            rows.extend({
                **base, "criterion_id": rating.criterion_id,
                "score": rating.score, "unable_to_judge": rating.unable_to_judge,
            } for rating in validated[0].ratings)
    return (
        pd.DataFrame(rows, columns=[
            "cohort", "document_id", "condition", "repetition", "task", "packet_id",
            "criterion_id", "score", "unable_to_judge",
        ]),
        pd.DataFrame(coverage),
        pd.DataFrame(duplicates, columns=[
            "cohort", "document_id", "condition", "repetition", "task", "packet_id", "result_path",
        ]),
    )


def paired_judge_comparison(ratings: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:
    """Compare only document/repetition/criterion units with every selected condition."""
    index = ["cohort", "document_id", "repetition", "task", "criterion_id"]
    usable = ratings.loc[~ratings["unable_to_judge"].astype(bool) & ratings["score"].notna()]
    wide = usable.pivot(index=index, columns="condition", values="score").reindex(columns=list(conditions))
    return wide.dropna(subset=list(conditions)).reset_index()
