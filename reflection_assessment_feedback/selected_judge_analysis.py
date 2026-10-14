from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from reflection_assessment_feedback.experiment_config import experiment_config_sha256, load_experiment_config
from reflection_assessment_feedback.judge_batch import build_judge_batch
from reflection_assessment_feedback.llm_judge import (
    FeedbackQualityJudgeOutput, JUDGE_OUTPUT_MODELS, _validate_span_comments,
    build_judge_input, judge_input_sha256,
)
from reflection_assessment_feedback.prompt_registry import resolve_prompt


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
        prompt = resolve_prompt(
            config["prompts"][f"{task}_llm_id"],
            config["prompts"][f"{task}_llm_version"],
        )
        canonical_gold = json.dumps(
            packet.get("gold_analysis_summary"), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
        expected = {
            "record_type": "llm_judge_quality_rating",
            "packet_id": identifier, "task": task,
            "evaluation_mode": "playground" if cohort == "playground" else "test",
            "evaluation_document_id": identity["document_id"],
            "evaluation_cohort_id": identity["cohort_id"],
            "source_artifact_sha256": identity["source_artifact_sha256"],
            "input_packet_sha256": judge_input_sha256(packet),
            "teacher_reference_sha256": hashlib.sha256(packet["reference_teacher_feedback"].encode()).hexdigest(),
            "gold_analysis_summary_sha256": (
                hashlib.sha256(canonical_gold.encode()).hexdigest()
                if task == "assessment_quality" else None
            ),
            "judge_provider": config["llm_judge"]["provider"],
            "judge_model": config["llm_judge"]["model_name"],
            "prompt_id": prompt.artifact_id, "prompt_version": prompt.version,
            "prompt_artifact_sha256": prompt.sha256,
            "experiment_config_sha256": experiment_config_sha256(),
        }
        validated = []
        for path, record in matches[identifier]:
            if any(record.get(field) != value for field, value in expected.items()):
                raise ValueError(f"Judge input or provenance mismatch: {path}")
            output = JUDGE_OUTPUT_MODELS[task].model_validate(record["result"])
            if isinstance(output, FeedbackQualityJudgeOutput):
                _validate_span_comments(output, build_judge_input(packet))
            validated.append(output)
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
