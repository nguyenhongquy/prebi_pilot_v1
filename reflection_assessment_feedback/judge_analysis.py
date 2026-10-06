from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from reflection_assessment_feedback.development_cohorts import (
    build_gold_segment_analysis_summary,
    load_remaining_test_feedback_cohort,
)
from reflection_assessment_feedback.experiment_config import (
    PROJECT_ROOT,
    experiment_config_sha256,
    load_experiment_config,
)
from reflection_assessment_feedback.llm_judge import JUDGE_OUTPUT_MODELS, judge_input_sha256
from reflection_assessment_feedback.prompt_registry import resolve_prompt

NEW_G2_LABEL = "G2 new direct GBERT"


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_new_g2_judge_scores(summary_path: Path, protected_root: Path) -> tuple[pd.DataFrame, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("condition") != "G2":
        raise ValueError("A completed G2 judge summary is required")
    protected_root = protected_root.resolve()
    batch = Path(summary["protected_batch"]).resolve()
    if not batch.is_relative_to(protected_root):
        raise ValueError("New G2 artifacts must remain within protected storage")
    provenance = json.loads((batch / "provenance.json").read_text(encoding="utf-8"))
    if summary["provenance"] != provenance:
        raise ValueError("Summary and protected batch provenance differ")
    config = load_experiment_config()
    checkpoint_directory = PROJECT_ROOT / config["paths"]["classifier_checkpoints"]
    if provenance.get("training_run_id") != checkpoint_directory.name:
        raise ValueError("New G2 does not use the configured checkpoint run")
    if provenance.get("classifier_architecture") != "direct" or provenance.get("canonical_authors") != 62:
        raise ValueError("Author-disjoint direct GBERT provenance is required")
    if provenance.get("experiment_config_sha256") != experiment_config_sha256():
        raise ValueError("New G2 config fingerprint differs from the current experiment")
    for name, digest in provenance["checkpoint_sha256"].items():
        if hashlib.sha256((checkpoint_directory / f"{name}.pt").read_bytes()).hexdigest() != digest:
            raise ValueError("New G2 checkpoint fingerprint mismatch")
    split_path = PROJECT_ROOT / config["paths"]["provisional_split_directory"] / "provisional_test_modeling_wide.csv"
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != provenance["test_split_sha256"]:
        raise ValueError("New G2 test split fingerprint mismatch")
    cohort = load_remaining_test_feedback_cohort()
    expected_ids = set(cohort["document_id"])
    if set(provenance["document_ids"]) != expected_ids or summary["documents"] != len(expected_ids):
        raise ValueError("New G2 does not cover the configured test cohort")
    references = cohort.set_index("document_id")["feedback_text"].to_dict()
    rows = []
    packets = {}
    identities = set()
    repetitions = config["generation"]["repetitions"]
    for path in sorted((batch / "judge").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        packet = json.loads((batch / "packets" / path.name).read_text(encoding="utf-8"))
        document_id = str(record["evaluation_document_id"])
        task = record["task"]
        repetition = record["repetition"]
        identity = (document_id, task, repetition)
        if identity in identities or document_id not in expected_ids or task not in JUDGE_OUTPUT_MODELS:
            raise ValueError("Duplicate or unexpected new G2 judge record")
        identities.add(identity)
        if record.get("evaluation_mode") != "test" or record.get("evaluation_condition") != "G2":
            raise ValueError("New G2 contains a non-test or non-G2 judgment")
        if record.get("judge_provider") != config["llm_judge"]["provider"] or record.get("judge_model") != config["llm_judge"]["model_name"]:
            raise ValueError("New G2 judge identity mismatch")
        prompt = resolve_prompt(config["prompts"][f"{task}_llm_id"], config["prompts"][f"{task}_llm_version"])
        if record.get("prompt_version") != prompt.version or record.get("prompt_artifact_sha256") != prompt.sha256:
            raise ValueError("New G2 judge prompt mismatch")
        if record.get("packet_id") != packet.get("packet_id") or packet.get("task") != task or not packet.get("condition_blinded"):
            raise ValueError("New G2 judge packet identity or blinding mismatch")
        if record.get("input_packet_sha256") != judge_input_sha256(packet):
            raise ValueError("New G2 judge input fingerprint mismatch")
        if packet["reference_teacher_feedback"] != references[document_id]:
            raise ValueError("New G2 teacher reference differs from the current cohort")
        if task == "assessment_quality" and packet.get("gold_analysis_summary") != build_gold_segment_analysis_summary(split_path, document_id):
            raise ValueError("New G2 gold-summary context mismatch")
        generation = json.loads((batch / "generation" / f"document-{document_id}-rep-{repetition}.json").read_text(encoding="utf-8"))
        if record.get("source_generation_sha256") != _fingerprint(generation) or generation["provenance"] != provenance:
            raise ValueError("New G2 generation provenance mismatch")
        if generation["run"]["model_name"] != config["generation"]["model_name"]:
            raise ValueError("New G2 generator identity mismatch")
        prediction = json.loads((batch / "predictions" / f"document-{document_id}.json").read_text(encoding="utf-8"))
        if generation["prediction_sha256"] != _fingerprint(prediction):
            raise ValueError("New G2 prediction provenance mismatch")
        output = JUDGE_OUTPUT_MODELS[task].model_validate(record["result"])
        packets[record["packet_id"]] = packet
        for rating in output.ratings:
            rows.append({
                "packet_id": record["packet_id"], "phase": "test", "document_id": document_id,
                "condition": NEW_G2_LABEL, "task": task, "criterion_id": rating.criterion_id,
                "score": rating.score, "unable_to_judge": rating.unable_to_judge,
                "repetition": repetition,
            })
    expected = {
        (document_id, task, repetition)
        for document_id in expected_ids
        for task in JUDGE_OUTPUT_MODELS
        for repetition in range(1, repetitions + 1)
    }
    if identities != expected or summary["judge_records"] != len(identities):
        raise ValueError("New G2 judge coverage is incomplete")
    return pd.DataFrame(rows), packets


def compare_new_g2_scores(historical: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["document_id", "task", "criterion_id"]
    selected = historical.loc[historical["phase"].eq("test") & historical["document_id"].isin(new["document_id"])].copy()
    selected["condition"] = selected["condition"].map({"G1": "G1 historical", "G2": "G2 historical", "G3": "G3 historical"})
    if selected["condition"].isna().any():
        raise ValueError("Unknown historical condition")
    comparison = pd.concat([selected, new], ignore_index=True)
    if comparison.duplicated([*keys, "condition"]).any():
        raise ValueError("Comparison requires one rating per document/condition/criterion; select one repetition")
    if set(selected["document_id"]) != set(new["document_id"]):
        raise ValueError("Historical and new G2 document cohorts differ")
    summary = comparison.groupby(["task", "criterion_id", "condition"], as_index=False).agg(
        document_count=("document_id", "nunique"), mean_score=("score", "mean"),
        median_score=("score", "median"), scorable_count=("score", "count"),
        unable_to_judge=("unable_to_judge", "sum"),
    )
    paired = comparison.pivot(index=keys, columns="condition", values="score")
    if NEW_G2_LABEL not in paired:
        raise ValueError("New G2 ratings are missing")
    differences = []
    for baseline in ("G1 historical", "G2 historical", "G3 historical"):
        if baseline not in paired:
            raise ValueError(f"Missing baseline: {baseline}")
        matched = paired[[NEW_G2_LABEL, baseline]].dropna().reset_index()
        matched["difference"] = matched[NEW_G2_LABEL] - matched[baseline]
        matched["comparison"] = f"{NEW_G2_LABEL} - {baseline}"
        differences.append(matched)
    paired_summary = pd.concat(differences).groupby(["task", "criterion_id", "comparison"], as_index=False).agg(
        paired_documents=("document_id", "nunique"), mean_difference=("difference", "mean"),
        median_difference=("difference", "median"),
        improved=("difference", lambda values: int(values.gt(0).sum())),
        unchanged=("difference", lambda values: int(values.eq(0).sum())),
        worsened=("difference", lambda values: int(values.lt(0).sum())),
    )
    return summary, paired_summary