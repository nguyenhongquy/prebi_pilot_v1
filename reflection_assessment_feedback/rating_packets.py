from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rubric_registry import resolve_rubric

RatingTask = Literal["assessment_quality", "feedback_quality"]

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


def build_development_rating_packet_bundle(
    protected_root: str | Path,
    *,
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
    feedback = pd.read_csv(
        feedback_path,
        usecols=["essay_name", "document_id", "feedback_text"],
        dtype={"essay_name": str, "document_id": str},
    )
    feedback["essay_name"] = feedback["essay_name"].str.strip().str.upper()
    feedback_by_document: dict[str, str] = {}
    feedback_rows = feedback.loc[
        feedback["essay_name"] == config["dataset"]["development_essay_name"]
    ]
    for document_id, document_feedback in feedback_rows.groupby("document_id"):
        feedback_values = {
            value.strip()
            for value in document_feedback["feedback_text"].dropna().astype(str)
            if value.strip()
        }
        if len(feedback_values) == 1:
            feedback_by_document[str(document_id)] = feedback_values.pop()
    r1_ids = sorted(
        set(feedback_by_document) & set(rows["document_id"].dropna())
    )
    if len(r1_ids) != 6:
        raise ValueError("Expected six R1 development documents for rating packet creation.")

    comparison_dir = root / "g1-g3" / "demo-runs"
    g2_dir = root / "g2" / "demo-runs"
    comparisons: dict[str, tuple[Path, dict[str, Any]]] = {}
    for artifact_path in sorted(comparison_dir.glob("comparison-*.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            artifact.get("development_cohort") == "R1"
            and artifact.get("document_id") in r1_ids
            and artifact.get("pairs")
        ):
            pair = artifact["pairs"][0]
            runs = [pair.get("G1", {}), pair.get("G3", {})]
            if all(
                run.get("prompt_version") == config["generation"]["expected_prompt_version"]
                and run.get("repetition") == config["generation"]["repetitions"]
                for run in runs
            ):
                comparisons[artifact["document_id"]] = (artifact_path, artifact)
    g2_runs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for artifact_path in sorted(g2_dir.glob("g2-run-*.json")):
        envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
        run = envelope.get("run", {})
        if (
            run.get("document_id") in r1_ids
            and run.get("condition") == "G2"
            and run.get("prompt_version") == config["generation"]["expected_prompt_version"]
            and run.get("rubric_version") == config["generation"]["expected_rubric_version"]
            and run.get("repetition") == config["generation"]["repetitions"]
        ):
            g2_runs[run["document_id"]] = (artifact_path, envelope)

    packets: list[dict[str, Any]] = []
    key_records: list[dict[str, Any]] = []
    for document_id in r1_ids:
        if document_id not in comparisons or document_id not in g2_runs:
            raise FileNotFoundError("A development document is missing a matched G1/G2/G3 artifact.")
        document_rows = rows.loc[rows["document_id"] == document_id].copy()
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
        comparison_path, comparison = comparisons[document_id]
        pair = comparison["pairs"][0]
        g2_path, g2_envelope = g2_runs[document_id]
        run_records = {
            "G1": pair["G1"],
            "G2": g2_envelope["run"],
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
    for document_id in r1_ids:
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
        "cohort": "R1-development",
        "records": key_records,
    }
    if len(packets) != 42 or score_packet_count != 6:
        raise ValueError("Expected 42 packets: 36 quality ratings and 6 feedback-implied score packets.")
    if write:
        output_dir = root / "rating-packets" / "development"
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(output_dir, 0o700)
        packet_path = output_dir / "r1-rating-packets.json"
        key_path = output_dir / "r1-condition-key.json"
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
