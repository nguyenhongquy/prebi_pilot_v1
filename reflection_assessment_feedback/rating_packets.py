from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from reflection_assessment_feedback.experiment_config import (
    experiment_config_sha256,
    load_experiment_config,
)
from reflection_assessment_feedback.development_cohorts import (
    REMAINING_TEST_COHORT,
    load_playground_feedback_cohort,
    load_remaining_test_feedback_cohort,
)
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rubric_registry import resolve_rubric

RatingTask = Literal["assessment_quality", "feedback_quality"]
PLAYGROUND_PACKET_COHORT = "playground-author-188-v1"
REMAINING_TEST_PACKET_COHORT = "remaining-test-author-disjoint-g2-v1"

LEGACY_RATING_CRITERIA: dict[str, list[dict[str, str]]] = {
    "assessment_quality": [
        {"criterion_id": "rubric_alignment", "label": "Rubric alignment", "description": "Scores reflect the rubric anchors and whole reflection."},
        {"criterion_id": "score_defensibility", "label": "Score defensibility", "description": "Scores are plausible given the reflection and rubric."},
        {"criterion_id": "evidence_support", "label": "Evidence support", "description": "Cited segments exist and support the judgments."},
        {"criterion_id": "justification_quality", "label": "Justification quality", "description": "Rationales are clear, dimension-specific, and evidence-linked."},
    ],
    "feedback_quality": [
        {"criterion_id": "correctness", "label": "Correctness", "description": "Claims fit the reflection and rubric without invented content."},
        {"criterion_id": "evidence_grounding", "label": "Evidence grounding", "description": "Claims are supported by relevant reflection evidence."},
        {"criterion_id": "specificity", "label": "Specificity", "description": "Feedback identifies relevant behavior or reasoning, not generic advice."},
        {"criterion_id": "actionability", "label": "Actionability", "description": "Next steps are concrete, feasible, and connected to an identified need."},
        {"criterion_id": "reflection_potential", "label": "Reflection potential", "description": "Feedback can help the learner reconsider assumptions, evidence, or alternatives."},
    ],
}

LEGACY_SCALE = {
    "min": 1,
    "max": 5,
    "anchors": {
        "1": "Very poor: materially wrong, unsupported, or unusable.",
        "2": "Poor: major problems; only limited useful content.",
        "3": "Mixed: partly useful, with noticeable issues.",
        "4": "Good: mostly sound and useful, with minor issues.",
        "5": "Excellent: consistently sound, specific, and useful.",
    },
}

RATING_CRITERIA = {
    "assessment_quality": [
        {
            "criterion_id": "rubric_alignment", "label": "Rubric alignment",
            "description": "Apply rubric dimensions and anchors to the whole reflection.",
            "anchors": {
                "1": "Materially misapplies rubric dimensions or anchors.",
                "2": "Partly aligns, with a limited inconsistency or unclear anchor application.",
                "3": "Consistently applies rubric dimensions and anchors to the whole reflection.",
            },
        },
        {
            "criterion_id": "score_defensibility", "label": "Score defensibility",
            "description": "Judge score plausibility, not agreement with hidden reference scores.",
            "anchors": {
                "1": "Scores are substantially implausible given the reflection and rubric.",
                "2": "Scores are partly defensible but a limited discrepancy remains.",
                "3": "Scores are plausible and defensible given the reflection and rubric.",
            },
        },
        {
            "criterion_id": "evidence_support", "label": "Evidence support",
            "description": "Check whether cited evidence supports the central judgments.",
            "anchors": {
                "1": "Central judgments rely on absent, irrelevant, or contradictory evidence.",
                "2": "Some evidence supports the judgments, but a meaningful gap remains.",
                "3": "Cited evidence exists and supports the central judgments.",
            },
        },
        {
            "criterion_id": "justification_quality", "label": "Justification quality",
            "description": "Judge the clarity and logic of dimension-specific rationales.",
            "anchors": {
                "1": "Rationales are missing, incoherent, or fail to explain the judgments.",
                "2": "Rationales partly explain the judgments but are vague or weakly connected.",
                "3": "Rationales clearly connect dimension judgments, rubric, and reflection evidence.",
            },
        },
    ],
    "feedback_quality": [
        {
            "criterion_id": "correctness", "label": "Correctness",
            "description": "Claims fit the reflection and rubric; citations are not required and omissions are not errors.",
            "anchors": {
                "1": "Major inaccuracies or misleading interpretations undermine the feedback.",
                "2": "Mostly accurate, but contains a limited error, overstatement, or questionable interpretation.",
                "3": "Accurate and defensible against the reflection and rubric, with no material inaccuracies.",
            },
        },
        {
            "criterion_id": "developmental_usefulness", "label": "Developmental usefulness",
            "description": "A relevant suggestion or productive reflective question can provide a meaningful way forward.",
            "anchors": {
                "1": "Generic, irrelevant, or provides no meaningful direction.",
                "2": "Relevant and partly useful, but the direction is vague or underdeveloped.",
                "3": "Offers a clear, relevant next step or productive direction for deeper reflection.",
            },
        },
    ],
}

SCALE = {"min": 1, "max": 3, "anchors": {"1": "Poor", "2": "Mixed", "3": "Strong"}}

ASSESSMENT_CRITERIA_V03 = [
    {
        "criterion_id": "score_rubric_fit", "label": "Score-Rubric Fit",
        "description": "Judge score plausibility against the reflection and rubric anchors, not hidden reference scores.",
        "anchors": {
            "1": "The score substantially conflicts with the reflection or rubric anchors.",
            "2": "The score is broadly plausible, but its placement within the rubric is questionable.",
            "3": "The score is defensible given the reflection and rubric anchors.",
        },
    },
    *RATING_CRITERIA["assessment_quality"][2:],
]


def _sha256_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        return item

    canonical = json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_assessment(run: dict[str, Any]) -> dict[str, Any]:
    raw_output = run.get("output", run)
    assessment = raw_output.get("assessment", {})
    if "component_assessments" in assessment:
        dimensions = assessment["component_assessments"]
        return {
            "dimensions": [
                {
                    "dimension_id": item["dimension_id"],
                    "score": item.get("band", item.get("score")),
                    "justification": item.get("justification", ""),
                    "evidence_segment_ids": item.get("evidence_segment_ids", []),
                }
                for item in dimensions
            ]
        }
    dimensions = assessment.get("dimensions", {})
    return {
        "dimensions": [
            {
                "dimension_id": dimension_id,
                "score": value["score"],
                "justification": value["justification"],
                "evidence_segment_ids": value.get("evidence_segment_ids", []),
            }
            for dimension_id, value in dimensions.items()
        ]
    }


def normalize_feedback(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_output = run.get("output", run)
    feedback = raw_output.get("feedback", {})
    normalized: dict[str, list[dict[str, Any]]] = {}
    for component in ("strengths", "weaknesses", "suggestions"):
        normalized[component] = [
            {
                "text": item.get("text", item.get("message", "")),
                "evidence_segment_ids": item.get("evidence_segment_ids", []),
            }
            for item in feedback.get(component, [])
        ]
    return normalized


def build_rating_packet(
    *,
    task: RatingTask,
    run: dict[str, Any],
    reflection_segments: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    if task not in RATING_CRITERIA:
        raise ValueError(f"Unsupported rating task: {task}")
    if task == "assessment_quality":
        protocol_config = load_experiment_config()["prompts"]
        protocol_id = protocol_config["assessment_quality_human_id"]
        protocol_version = protocol_config["assessment_quality_human_version"]
        protocol = resolve_prompt(protocol_id, protocol_version)
        output = normalize_assessment(run)
        displayed_context = ["segmented_reflection", "rubric", "assessment_output"]
    else:
        protocol_config = load_experiment_config()["prompts"]
        protocol_id = protocol_config["feedback_quality_human_id"]
        protocol_version = protocol_config["feedback_quality_human_version"]
        protocol = resolve_prompt(protocol_id, protocol_version)
        output = normalize_feedback(run)
        displayed_context = ["segmented_reflection", "rubric", "feedback_output"]

    packet_id = str(uuid4())
    supported_versions = {"0.1.0", "0.2.0", "0.3.0"} if task == "assessment_quality" else {"0.1.0", "0.2.0"}
    if protocol_version not in supported_versions:
        raise ValueError("Unsupported quality protocol version; define its criteria and scale first.")
    legacy = protocol_version == "0.1.0"
    criteria = LEGACY_RATING_CRITERIA[task] if legacy else RATING_CRITERIA[task]
    if task == "assessment_quality" and protocol_version == "0.3.0":
        criteria = ASSESSMENT_CRITERIA_V03
    reflection_snapshot = {"segments": reflection_segments}
    packet = {
        "packet_id": packet_id,
        "task": task,
        "packet_version": "1.0.0",
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "human_protocol": protocol.text,
        "criteria": criteria,
        "scale": LEGACY_SCALE if legacy else SCALE,
        "display_output_sha256": _sha256_json(output),
        "displayed_reflection_sha256": _sha256_json(reflection_snapshot),
        "rubric_version": rubric["version"],
        "reflection": reflection_snapshot,
        "rubric": rubric,
        "output": output,
        "condition_blinded": True,
        "displayed_context": displayed_context,
    }
    return packet


def _select_packet_cohort(
    cohort: Literal["playground", "remaining-test"],
    *,
    config: dict[str, Any],
) -> tuple[str, list[str], dict[str, str]]:
    if cohort == "playground":
        cohort_feedback = load_playground_feedback_cohort()
        selected_document_ids = cohort_feedback["document_id"].astype(str).tolist()
        selected_feedback = dict(
            zip(
                selected_document_ids,
                cohort_feedback["feedback_text"].astype(str),
                strict=True,
            )
        )
        return PLAYGROUND_PACKET_COHORT, selected_document_ids, selected_feedback

    cohort_feedback = load_remaining_test_feedback_cohort()
    selected_document_ids = cohort_feedback["document_id"].astype(str).tolist()
    feedback_by_document = dict(
        zip(
            selected_document_ids,
            cohort_feedback["feedback_text"].astype(str),
            strict=True,
        )
    )
    return REMAINING_TEST_PACKET_COHORT, selected_document_ids, feedback_by_document


def _artifact_fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_author_disjoint_g2_runs(
    protected_root: Path,
    *,
    cohort: Literal["playground", "remaining-test"],
    document_ids: list[str],
    config: dict[str, Any],
    project_root: Path,
    rubric_sha256: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    checkpoint_directory = project_root / config["paths"]["classifier_checkpoints"]
    checkpoint_manifest = json.loads((checkpoint_directory / "run-manifest.json").read_text())
    expected_rule = (
        {"playground_author_document_id": str(config["dataset"]["exploratory_document_id"])}
        if cohort == "playground"
        else {
            "excluded_author_document_id": str(config["dataset"]["exploratory_document_id"]),
            "excluded_document_ids": config["dataset"]["generation_excluded_document_ids"],
        }
    )
    expected_ids = set(document_ids)
    matching_batches: list[tuple[Path, dict[str, Any]]] = []
    for provenance_path in sorted(
        (protected_root / "g2" / "author-disjoint-evaluation").glob("*/provenance.json")
    ):
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            set(map(str, provenance.get("document_ids", []))) == expected_ids
            and provenance.get("cohort_rule") == expected_rule
            and provenance.get("training_run_id") == checkpoint_directory.name
            and provenance.get("classifier_architecture") == "direct"
            and provenance.get("classifier_backend") == "gbert"
            and provenance.get("canonical_authors") == 62
            and provenance.get("experiment_config_sha256") == experiment_config_sha256()
            and (cohort != "playground" or provenance.get("langsmith_tracing") is False)
        ):
            batch = provenance_path.parent
            complete = all(
                (batch / "predictions" / f"document-{document_id}.json").is_file()
                and (
                    batch
                    / "generation"
                    / f"document-{document_id}-rep-{config['generation']['repetitions']}.json"
                ).is_file()
                for document_id in document_ids
            )
            if complete:
                matching_batches.append((batch, provenance))
    if len(matching_batches) != 1:
        raise FileNotFoundError(
            f"Expected one complete current author-disjoint G2 batch for {cohort}; found {len(matching_batches)}."
        )

    batch, provenance = matching_batches[0]
    direct_model_names = (
        "scope",
        "direct-situationserfassung",
        "direct-analyse",
        "direct-konsequenzen",
    )
    for model_name in direct_model_names:
        model_record = checkpoint_manifest["models"][model_name]
        checkpoint_path = checkpoint_directory / f"{model_name}.pt"
        if (
            model_name not in provenance["checkpoint_sha256"]
            or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            != provenance["checkpoint_sha256"][model_name]
            or provenance["checkpoint_sha256"][model_name]
            != model_record["checkpoint_sha256"]
        ):
            raise ValueError("Author-disjoint G2 checkpoint provenance is invalid.")
    for split_name, expected_hash in checkpoint_manifest["split_sha256"].items():
        split, view = split_name.split("_", 1)
        split_path = (
            project_root
            / config["paths"]["provisional_split_directory"]
            / f"provisional_{split}_{view}.csv"
        )
        if hashlib.sha256(split_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Author-disjoint G2 split provenance is invalid.")

    generation_prompt = resolve_prompt(
        config["generation"]["prompt_id"],
        config["generation"]["expected_prompt_version"],
    )
    runs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for document_id in document_ids:
        prediction_path = batch / "predictions" / f"document-{document_id}.json"
        if not prediction_path.is_file():
            raise FileNotFoundError("Author-disjoint G2 prediction is missing from its batch.")
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        if any(prediction.get(key) != value for key, value in provenance.items()):
            raise ValueError("Author-disjoint G2 prediction has mismatched batch provenance.")

        generation_path = (
            batch
            / "generation"
            / f"document-{document_id}-rep-{config['generation']['repetitions']}.json"
        )
        if not generation_path.is_file():
            raise FileNotFoundError("Author-disjoint G2 generation is incomplete for the selected cohort.")
        envelope = json.loads(generation_path.read_text(encoding="utf-8"))
        run = envelope.get("run", {})
        if (
            envelope.get("provenance") != provenance
            or envelope.get("document_id") != document_id
            or envelope.get("prediction_sha256") != _artifact_fingerprint(prediction)
            or run.get("document_id", envelope["document_id"]) != document_id
            or run.get("condition") != "G2"
            or run.get("model_name") != config["generation"]["model_name"]
            or run.get("prompt_version") != generation_prompt.version
            or run.get("prompt_artifact_sha256") != generation_prompt.sha256
            or run.get("rubric_artifact_sha256") != rubric_sha256
            or run.get("repetition") != config["generation"]["repetitions"]
            or run.get("generation_parameters")
            != {"temperature": config["generation"]["temperature"]}
        ):
            raise ValueError("Author-disjoint G2 generation provenance or configuration is invalid.")
        runs[document_id] = (generation_path, run)
    return runs


def build_development_rating_packet_bundle(
    protected_root: str | Path,
    *,
    cohort: Literal["playground", "remaining-test"] = "remaining-test",
    write: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_experiment_config()
    root = Path(protected_root).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    if root == project_root or project_root in root.parents:
        raise ValueError("Protected rating packet storage must be outside the source tree.")

    rubric_entry = resolve_rubric(
        config["generation"]["rubric_id"],
        config["generation"]["expected_rubric_version"],
    )
    rubric_json = rubric_entry.rubric.model_dump(mode="json")
    split_path = (
        project_root
        / config["paths"]["provisional_split_directory"]
        / f"provisional_{config['dataset']['prediction_split']}_modeling_wide.csv"
    )
    import pandas as pd

    rows = pd.read_csv(
        split_path,
        usecols=["document_id", "segment_id", "text", "segment_order"],
        dtype={"document_id": str, "segment_id": str},
    )
    feedback_path = (project_root / config["paths"]["human_feedback"]).resolve()
    cohort_id, selected_document_ids, feedback_by_document = _select_packet_cohort(
        cohort,
        config=config,
    )
    if not selected_document_ids or len(set(selected_document_ids)) != len(selected_document_ids):
        raise ValueError("Selected rating cohort is empty or contains duplicate document IDs.")
    if not set(selected_document_ids).issubset(set(rows["document_id"].dropna().astype(str))):
        raise ValueError("A selected rating document is missing from the configured test split.")

    comparison_dir = root / "g1-g3" / "demo-runs"
    comparison_candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for artifact_path in sorted(comparison_dir.glob("comparison-*.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            str(artifact.get("document_id")) in selected_document_ids
            and artifact.get("rubric_id") == config["generation"]["rubric_id"]
            and artifact.get("rubric_version") == config["generation"]["expected_rubric_version"]
            and artifact.get("pairs")
        ):
            pair = artifact["pairs"][0]
            runs = [pair.get("G1", {}), pair.get("G3", {})]
            if all(
                run.get("condition") in {"G1", "G3"}
                and run.get("model_name") == config["generation"]["model_name"]
                and run.get("prompt_version") == config["generation"]["expected_prompt_version"]
                and run.get("repetition") == config["generation"]["repetitions"]
                and run.get("generation_parameters")
                == {"temperature": config["generation"]["temperature"]}
                for run in runs
            ):
                comparison_candidates.setdefault(str(artifact["document_id"]), []).append(
                    (artifact_path, artifact)
                )
    if set(comparison_candidates) != set(selected_document_ids):
        raise FileNotFoundError("A selected document lacks a compatible G1/G3 comparison artifact.")
    author_disjoint_runs = _load_author_disjoint_g2_runs(
        root,
        cohort=cohort,
        document_ids=selected_document_ids,
        config=config,
        project_root=project_root,
        rubric_sha256=rubric_entry.sha256,
    )

    matched_artifacts: dict[str, tuple[Path, dict[str, Any], Path, dict[str, Any]]] = {}
    for document_id in selected_document_ids:
        comparison_path, comparison = max(
            comparison_candidates[document_id], key=lambda candidate: candidate[0].name
        )
        g2_path, g2_run = author_disjoint_runs[document_id]
        matched_artifacts[document_id] = (comparison_path, comparison, g2_path, g2_run)

    packets: list[dict[str, Any]] = []
    key_records: list[dict[str, Any]] = []
    for document_id in selected_document_ids:
        document_rows = rows.loc[rows["document_id"] == document_id].copy()
        if document_rows.empty:
            raise ValueError(f"Selected rating document {document_id} has no reflection segments.")
        consistency = document_rows.groupby("segment_id").agg(
            text_variants=("text", "nunique"),
            order_variants=("segment_order", "nunique"),
        )
        if (consistency > 1).any().any():
            raise ValueError("Reflection candidate copies disagree on segment text or order.")
        document_rows = document_rows.drop_duplicates("segment_id").sort_values("segment_order")
        reflection_segments = [
            {
                "segment_id": row.segment_id,
                "order": int(row.segment_order),
                "text": str(row.text),
            }
            for row in document_rows.itertuples(index=False)
        ]
        comparison_path, comparison, g2_path, g2_envelope = matched_artifacts[document_id]
        pair = comparison["pairs"][0]
        run_records = {
            "G1": pair["G1"],
            "G2": g2_run,
            "G3": pair["G3"],
        }
        for condition, run in run_records.items():
            if run.get("condition") != condition or run.get("document_id", document_id) != document_id:
                raise ValueError("A saved generation run has mismatched condition or document provenance.")
            for task in ("assessment_quality", "feedback_quality"):
                packet = build_rating_packet(
                    task=task,
                    run=run,
                    reflection_segments=reflection_segments,
                    rubric=rubric_json,
                )
                packets.append(packet)
                key_records.append(
                    {
                        "packet_id": packet["packet_id"],
                        "document_id": document_id,
                        "condition": condition,
                        "run_id": run["run_id"],
                        "repetition": run["repetition"],
                        "task": task,
                        "source_artifact_sha256": hashlib.sha256(
                            (g2_path if condition == "G2" else comparison_path).read_bytes()
                        ).hexdigest(),
                    }
                )

    score_protocol_config = config["prompts"]
    score_protocol = resolve_prompt(
        score_protocol_config["feedback_implied_score_human_id"],
        score_protocol_config["feedback_implied_score_human_version"],
    )
    score_packet_count = 0
    for document_id in selected_document_ids:
        human_feedback = feedback_by_document[document_id]
        reflection_segments = [
            {
                "segment_id": row.segment_id,
                "order": int(row.segment_order),
                "text": str(row.text),
            }
            for row in (
                rows.loc[rows["document_id"] == document_id]
                .drop_duplicates("segment_id")
                .sort_values("segment_order")
                .itertuples(index=False)
            )
        ]
        reflection_snapshot = {"segments": reflection_segments}
        packet = {
            "packet_id": str(uuid4()),
            "task": "feedback_implied_score",
            "packet_version": "1.0.0",
            "protocol_id": score_protocol.artifact_id,
            "protocol_version": score_protocol.version,
            "human_protocol": score_protocol.text,
            "score_dimensions": [
                {"dimension_id": dimension["dimension_id"], "label": dimension["name"]}
                for dimension in rubric_json["dimensions"]
            ],
            "display_output_sha256": _sha256_json(human_feedback),
            "displayed_reflection_sha256": _sha256_json(reflection_snapshot),
            "rubric_version": rubric_entry.version,
            "reflection": reflection_snapshot,
            "rubric": rubric_json,
            "human_feedback": human_feedback,
            "condition_blinded": True,
            "displayed_context": ["segmented_reflection", "rubric", "original_human_feedback"],
        }
        packets.append(packet)
        key_records.append(
            {
                "packet_id": packet["packet_id"],
                "document_id": document_id,
                "condition": None,
                "run_id": None,
                "repetition": None,
                "task": "feedback_implied_score",
                "source_artifact_sha256": hashlib.sha256(feedback_path.read_bytes()).hexdigest(),
            }
        )
        score_packet_count += 1

    packet_bundle = {"schema_version": "1.0.0", "packets": packets}
    condition_key = {
        "schema_version": "1.0.0",
        "cohort": cohort_id,
        "records": key_records,
    }
    expected_packet_count = len(selected_document_ids) * 7
    if len(packets) != expected_packet_count or score_packet_count != len(selected_document_ids):
        raise ValueError("Rating packet count differs from the configured cohort size.")
    if write:
        output_dir = root / "rating-packets" / "development"
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(output_dir, 0o700)
        safe_cohort_name = cohort_id.lower().replace("_", "-")
        packet_path = output_dir / f"{safe_cohort_name}-rating-packets.json"
        key_path = output_dir / f"{safe_cohort_name}-condition-key.json"
        if packet_path.exists() or key_path.exists():
            raise FileExistsError("Refusing to overwrite existing rating packet exports.")
        for path, content in ((packet_path, packet_bundle), (key_path, condition_key)):
            temporary_path = None
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=output_dir,
                    prefix=".rating-packets-", suffix=".tmp", delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    json.dump(content, temporary_file, ensure_ascii=False, indent=2)
                    temporary_file.write("\n")
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
    return packet_bundle, condition_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded R1 rating packets in protected storage.")
    parser.add_argument("--protected-root", type=Path, default=os.environ.get("PREBI_DATA_ROOT"))
    parser.add_argument("--write", action="store_true", help="Write packet bundle and separate condition key.")
    arguments = parser.parse_args()
    if arguments.protected_root is None:
        parser.error("Set PREBI_DATA_ROOT or pass --protected-root.")
    packets, _key = build_development_rating_packet_bundle(arguments.protected_root, write=arguments.write)
    print(f"Validated {len(packets['packets'])} blinded packets; written={arguments.write}.")


if __name__ == "__main__":
    main()
