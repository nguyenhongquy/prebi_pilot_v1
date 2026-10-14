from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from reflection_assessment_feedback.data import load_human_analysis_context
from reflection_assessment_feedback.development_cohorts import (
    build_gold_segment_analysis_summary,
    load_playground_feedback_cohort,
    load_remaining_test_feedback_cohort,
)
from reflection_assessment_feedback.experiment_config import (
    PROJECT_ROOT, experiment_config_sha256, load_experiment_config,
)
from reflection_assessment_feedback.llm_judge import build_judge_input
from reflection_assessment_feedback.models import GenerationRequest, IntermediateAnalysis
from reflection_assessment_feedback.prompt import build_prompt, prompt_components_for_request
from reflection_assessment_feedback.rating_packets import (
    build_rating_packet,
)
from reflection_assessment_feedback.rubric_registry import resolve_rubric


def build_judge_batch(
    protected_root: str | Path,
    *,
    cohort: Literal["playground", "remaining-test"] = "playground",
    conditions: tuple[str, ...] = ("G1", "G3"),
    tasks: tuple[Literal["assessment_quality", "feedback_quality"], ...] = ("assessment_quality", "feedback_quality"),
    pilot_only: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build stable, blinded judge inputs directly from compatible saved generations."""
    if cohort not in {"playground", "remaining-test"}:
        raise ValueError("Cohort must be playground or remaining-test.")
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - {"G1", "G2", "G3"}:
        raise ValueError("Select unique conditions from G1, G2, G3.")
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"assessment_quality", "feedback_quality"}:
        raise ValueError("Select unique assessment_quality and/or feedback_quality tasks.")
    root = Path(protected_root).expanduser().resolve()
    if root == PROJECT_ROOT or PROJECT_ROOT in root.parents:
        raise ValueError("Protected judge inputs must be outside the source tree.")
    config = load_experiment_config()
    generation = config["generation"]
    rubric = resolve_rubric(generation["rubric_id"], generation["expected_rubric_version"])
    feedback = load_playground_feedback_cohort() if cohort == "playground" else load_remaining_test_feedback_cohort()
    records = feedback.to_dict(orient="records")
    if not records or feedback["document_id"].astype(str).duplicated().any():
        raise ValueError("Selected cohort must contain unique documents.")
    if cohort == "remaining-test" and len(records) != config["dataset"]["expected_generation_document_count"]:
        raise ValueError("Remaining-test cohort size differs from configuration.")
    if pilot_only:
        records = records[:1]
    document_ids = [str(record["document_id"]) for record in records]
    cohort_id = "playground-author-188-v1" if cohort == "playground" else config["dataset"]["generation_cohort"]
    split_path = PROJECT_ROOT / config["paths"]["provisional_split_directory"] / f"provisional_{config['dataset']['prediction_split']}_modeling_wide.csv"
    config_hash = experiment_config_sha256()
    comparisons: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    if set(conditions) & {"G1", "G3"}:
        for path in sorted((root / "g1-g3" / "demo-runs").glob("comparison-*.json")):
            artifact = json.loads(path.read_text())
            document_id = str(artifact.get("document_id"))
            if document_id not in document_ids:
                continue
            if (
                artifact.get("development_cohort") == cohort_id
                and artifact.get("experiment_config_sha256") == config_hash
                and artifact.get("rubric_version") == rubric.version
                and artifact.get("rubric_artifact_sha256") == rubric.sha256
            ):
                comparisons.setdefault(document_id, []).append((path, artifact))
        for document_id in document_ids:
            if len(comparisons.get(document_id, [])) != 1:
                raise ValueError(f"Expected one compatible G1/G3 artifact for document {document_id}; archive duplicate current runs or generate missing runs.")
    g2_runs: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = {}
    g2_predictions = {}
    if "G2" in conditions:
        comparison_hashes = {}
        for path in (root / "g1-g3" / "demo-runs").glob("comparison-*.json"):
            artifact = json.loads(path.read_text())
            if (
                artifact.get("development_cohort") == cohort_id
                and artifact.get("experiment_config_sha256") == config_hash
                and artifact.get("rubric_version") == rubric.version
            ):
                comparison_hashes[hashlib.sha256(path.read_bytes()).hexdigest()] = str(artifact["document_id"])
        for document_id in document_ids:
            prediction_path = root / "g2" / "predictions" / f"predicted-analysis-{quote(document_id, safe='')}.json"
            prediction = json.loads(prediction_path.read_text())
            if (
                prediction.get("document_id") != document_id
                or prediction.get("experiment_config_sha256") != config_hash
            ):
                raise ValueError("G2 prediction document or configuration fingerprint differs from current inputs.")
            g2_predictions[document_id] = (
                hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                IntermediateAnalysis.model_validate(prediction["analysis"]),
            )
            if g2_predictions[document_id][1].source != "predicted":
                raise ValueError("G2 requires predicted analysis.")
        for path in sorted((root / "g2" / "demo-runs").glob("g2-run-*.json")):
            envelope = json.loads(path.read_text())
            run = envelope.get("run", {})
            document_id = str(run.get("document_id"))
            if document_id not in document_ids:
                continue
            if (
                envelope.get("record_type") == "g2_generation_run"
                and envelope.get("experiment_config_sha256") == config_hash
                and comparison_hashes.get(envelope.get("paired_g1_g3_comparison_sha256")) == document_id
                and run.get("analysis_reference_id") == f"predicted-analysis-sha256-{g2_predictions[document_id][0]}"
            ):
                g2_runs.setdefault((document_id, run.get("repetition")), []).append((path, run))
        for document_id in document_ids:
            for repetition in range(1, generation["repetitions"] + 1):
                if len(g2_runs.get((document_id, repetition), [])) != 1:
                    raise ValueError(
                        f"Expected one compatible notebook G2 artifact for document {document_id}, repetition {repetition}; "
                        "check prediction/configuration hashes and paired G1/G3 artifacts, or archive duplicate current runs."
                    )
    packets = []
    key = {}
    for record in records:
        document_id = str(record["document_id"])
        document, human = load_human_analysis_context(split_path, document_id)
        segments = [segment.model_dump(mode="json") for segment in document.segments]
        summary = build_gold_segment_analysis_summary(split_path, document_id) if "assessment_quality" in tasks else None
        runs = {}
        if set(conditions) & {"G1", "G3"}:
            path, artifact = comparisons[document_id][0]
            if "G3" in conditions and artifact.get("g3_human_analysis_context") != human.model_dump(mode="json"):
                raise ValueError("Saved G3 annotations differ from the current source.")
            pairs = artifact.get("pairs", [])
            if len(pairs) != generation["repetitions"]:
                raise ValueError("G1/G3 repetition coverage differs from configuration.")
            if {pair["repetition"] for pair in pairs} != set(range(1, generation["repetitions"] + 1)):
                raise ValueError("G1/G3 repetition identifiers must be unique and complete.")
            for pair in pairs:
                repetition = pair["repetition"]
                for condition in set(conditions) & {"G1", "G3"}:
                    runs[(condition, repetition)] = (path, pair[condition])
        if "G2" in conditions:
            for repetition in range(1, generation["repetitions"] + 1):
                runs[("G2", repetition)] = g2_runs[(document_id, repetition)][0]
        for condition in conditions:
            for repetition in range(1, generation["repetitions"] + 1):
                path, run = runs[(condition, repetition)]
                if (
                    run.get("condition") != condition
                    or run.get("repetition") != repetition
                    or run.get("model_name") != generation["model_name"]
                    or run.get("prompt_version") != generation["expected_prompt_version"]
                    or run.get("rubric_artifact_sha256") != rubric.sha256
                    or run.get("generation_parameters") != {}
                ):
                    raise ValueError("Saved generation provenance differs from configuration.")
                if condition in {"G1", "G2", "G3"}:
                    request = GenerationRequest(
                        condition="G1" if condition == "G1" else "G2" if condition == "G2" else "G3",
                        document=document, rubric=rubric.rubric,
                        analysis=(
                            human if condition == "G3"
                            else g2_predictions[document_id][1] if condition == "G2"
                            else None
                        ),
                    )
                    expected_components = [
                        {"artifact_id": a.artifact_id, "version": a.version, "sha256": a.sha256}
                        for a in prompt_components_for_request(request)
                    ]
                    if (
                        run.get("prompt_components") != expected_components
                        or run.get("prompt_artifact_sha256") != expected_components[0]["sha256"]
                        or run.get("rendered_prompt_sha256") != hashlib.sha256(build_prompt(request).encode()).hexdigest()
                    ):
                        raise ValueError("Saved generation prompt or reflection snapshot differs from current inputs.")
                source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                for task in tasks:
                    packet = build_rating_packet(task=task, run=run, reflection_segments=segments, rubric=rubric.rubric.model_dump(mode="json"))
                    packet["reference_teacher_feedback"] = record["feedback_text"]
                    if task == "assessment_quality":
                        packet["gold_analysis_summary"] = summary
                    identity = json.dumps({
                        "input": build_judge_input(packet), "source": source_hash,
                        "document": document_id, "condition": condition, "repetition": repetition, "task": task,
                    }, sort_keys=True, ensure_ascii=False)
                    packet["packet_id"] = str(uuid5(NAMESPACE_URL, identity))
                    packets.append(packet)
                    key[packet["packet_id"]] = {
                        "document_id": document_id, "condition": condition, "repetition": repetition,
                        "cohort_id": cohort_id, "source_artifact_sha256": source_hash,
                        "run_id": run["run_id"], "task": task,
                    }
    return packets, key
