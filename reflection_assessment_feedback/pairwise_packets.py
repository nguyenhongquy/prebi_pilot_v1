from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

import pandas as pd

from reflection_assessment_feedback.experiment_config import (
    experiment_config_sha256,
    load_experiment_config,
)
from reflection_assessment_feedback.prompt_registry import MANIFEST_PATH, resolve_prompt
from reflection_assessment_feedback.rubric_registry import resolve_rubric
from reflection_assessment_feedback.rating_packets import (
    _load_author_disjoint_g2_runs,
    _select_packet_cohort,
    build_segment_annotation_summary,
    build_segment_modelling_annotations,
    REFERENCE_SCORE_PROTOCOL_ID,
    REFERENCE_SCORE_PROTOCOL_VERSION,
    make_reflection_id,
    normalize_feedback,
)

SCHEMA_VERSION = "1.2.0"
PROTOCOL_ID = "prebi_blinded_feedback_preference"
Condition = Literal["G1", "G2", "G3"]
CONDITIONS: tuple[Condition, ...] = ("G1", "G2", "G3")
EVALUATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
EXPOSED_CONDITION_PATTERN = re.compile(r"(?i)(?<![a-z0-9])(?:g[1-3]|gemini|gpt|claude)(?![a-z0-9])")


@lru_cache(maxsize=None)
def _latest_active_prompt_version(artifact_id: str) -> str:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    versions = {
        item["version"]
        for item in manifest["artifacts"]
        if item.get("id") == artifact_id and item.get("status") == "active"
    }
    if not versions:
        raise ValueError(f"No active prompt is registered for {artifact_id!r}.")
    return max(versions, key=lambda version: tuple(int(part) for part in version.split(".")))


def _expected_prompt_components(
    config: dict[str, Any],
    condition: Condition,
) -> list[dict[str, str]]:
    generation = config["generation"]
    entries = [
        resolve_prompt(
            generation["prompt_id"],
            generation["expected_prompt_version"],
        )
    ]
    if condition == "G2":
        entries.append(
            resolve_prompt(
                generation["g2_supplemental_prompt_id"],
                generation["g2_supplemental_prompt_version"],
            )
        )
    elif condition == "G3":
        entries.append(
            resolve_prompt(
                generation["g3_supplemental_prompt_id"],
                generation["g3_supplemental_prompt_version"],
            )
        )
    if any(entry.status != "active" for entry in entries):
        raise ValueError("Pairwise generation requires active prompt artifacts; draft prompts are not accepted.")
    if any(entry.version != _latest_active_prompt_version(entry.artifact_id) for entry in entries):
        raise ValueError("Experiment config does not pin the latest active generation prompt versions.")
    return [
        {
            "artifact_id": entry.artifact_id,
            "version": entry.version,
            "sha256": entry.sha256,
        }
        for entry in entries
    ]


def _validate_run_provenance(
    run: dict[str, Any],
    *,
    condition: Condition,
    config: dict[str, Any],
    rubric_sha256: str,
) -> list[dict[str, str]]:
    generation = config["generation"]
    expected_components = _expected_prompt_components(config, condition)
    if (
        run.get("condition") != condition
        or run.get("model_name") != generation["model_name"]
        or run.get("prompt_version") != generation["expected_prompt_version"]
        or run.get("prompt_artifact_sha256") != expected_components[0]["sha256"]
        or run.get("prompt_components") != expected_components
        or run.get("rubric_version") != generation["expected_rubric_version"]
        or run.get("rubric_artifact_sha256") != rubric_sha256
        or run.get("pipeline_version") != generation["pipeline_version"]
        or run.get("repetition") != generation["repetitions"]
        or run.get("generation_parameters") != {}
    ):
        raise ValueError(
            f"{condition} generation does not match the latest active prompt, rubric, "
            "pipeline, and model configuration."
        )
    return expected_components


def _load_latest_g1_g3_runs(
    protected_root: Path,
    *,
    document_ids: Sequence[str],
    config: dict[str, Any],
    rubric_sha256: str,
) -> dict[str, tuple[Path, dict[str, Any], dict[str, Any]]]:
    comparison_dir = protected_root / "g1-g3" / "demo-runs"
    if not comparison_dir.is_dir():
        raise FileNotFoundError(f"G1/G3 comparison directory is missing: {comparison_dir}")

    selected = set(document_ids)
    candidates: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {}
    generation = config["generation"]
    for artifact_path in sorted(comparison_dir.glob("comparison-*.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        document_id = str(artifact.get("document_id", ""))
        if document_id not in selected:
            continue
        if (
            artifact.get("rubric_id") != generation["rubric_id"]
            or artifact.get("rubric_version") != generation["expected_rubric_version"]
        ):
            continue

        matching_pairs = [
            pair for pair in artifact.get("pairs", [])
            if pair.get("repetition") == generation["repetitions"]
        ]
        if len(matching_pairs) != 1:
            continue
        pair = matching_pairs[0]
        runs = {"G1": pair.get("G1", {}), "G3": pair.get("G3", {})}
        try:
            for condition, run in runs.items():
                _validate_run_provenance(
                    run,
                    condition=condition,
                    config=config,
                    rubric_sha256=rubric_sha256,
                )
                if str(run.get("document_id", document_id)) != document_id:
                    raise ValueError("G1/G3 generation document provenance does not match its artifact.")
        except ValueError:
            continue
        candidates.setdefault(document_id, []).append((artifact_path, artifact, pair))

    missing = sorted(selected - set(candidates))
    if missing:
        raise FileNotFoundError(
            "No G1/G3 comparison matches the configured latest active prompt and "
            f"rubric for document IDs: {', '.join(missing)}."
        )
    return {
        document_id: max(candidates[document_id], key=lambda item: item[0].name)
        for document_id in document_ids
    }


def _reflection_segments(document_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if document_rows.empty:
        raise ValueError("A selected evaluation case has no reflection segments.")
    consistency = document_rows.groupby("segment_id").agg(
        text_variants=("text", "nunique"),
        order_variants=("segment_order", "nunique"),
    )
    if (consistency > 1).any().any():
        raise ValueError("Reflection candidate copies disagree on segment text or order.")
    if document_rows["candidate_id"].isna().any() or document_rows["candidate_id"].str.strip().eq("").any():
        raise ValueError("Human annotation candidates contain missing candidate IDs.")
    if document_rows.duplicated(["segment_id", "candidate_id"]).any():
        raise ValueError("Human annotations contain duplicate segment/candidate rows.")

    unique_segments = document_rows.drop_duplicates("segment_id").sort_values("segment_order")
    segments = [
        {
            "segment_id": str(row.segment_id),
            "order": int(row.segment_order),
            "text": str(row.text),
        }
        for row in unique_segments.itertuples(index=False)
    ]
    segment_ids = [segment["segment_id"] for segment in segments]
    if (
        not segments
        or len(set(segment_ids)) != len(segment_ids)
        or any(not segment_id.strip() for segment_id in segment_ids)
        or any(not segment["text"].strip() for segment in segments)
        or len({segment["order"] for segment in segments}) != len(segments)
    ):
        raise ValueError("Reflection segments must have unique IDs, text, and order values.")
    return segments


def _human_annotations(
    document_rows: pd.DataFrame,
    rubric: dict[str, Any],
) -> list[dict[str, str]]:
    annotations = build_segment_modelling_annotations(document_rows)
    dimensions = {
        item["dimension_id"]: item["name"]
        for item in rubric["dimensions"]
    }
    visible: set[tuple[str, str, str, str]] = set()
    for item in annotations:
        dimension_id = item["dimension_id"]
        level = item["level"]
        if item["known"] is not True or level not in {"0", "1", "2", "3"}:
            continue
        if dimension_id not in dimensions:
            raise ValueError(f"Human annotation uses unknown rubric dimension {dimension_id!r}.")
        review_note = " (prüfbedürftig)" if item["requires_review"] else ""
        visible.add(
            (
                item["segment_id"],
                dimension_id,
                dimensions[dimension_id],
                f"Band {level}{review_note}",
            )
        )
    result = [
        {
            "segment_id": segment_id,
            "dimension_id": dimension_id,
            "skill_label": skill_label,
            "level": level,
        }
        for segment_id, dimension_id, skill_label, level in sorted(visible)
    ]
    if not result:
        raise ValueError("No known human segment annotations are available for the evaluation case.")
    return result


def _candidate_feedback(
    run: dict[str, Any],
    *,
    segment_ids: set[str],
    condition: str,
) -> dict[str, list[dict[str, Any]]]:
    feedback = normalize_feedback(run)
    entries = [entry for group in feedback.values() for entry in group]
    if not entries:
        raise ValueError(f"{condition} generation contains no feedback statements.")
    for entry in entries:
        if not isinstance(entry["text"], str) or not entry["text"].strip():
            raise ValueError(f"{condition} feedback statements must contain nonempty text.")
        evidence_ids = entry["evidence_segment_ids"]
        # Empty evidence is kept: statements about absent content may cite no segment;
        # the evaluator UI shows "Keine Belege angegeben".
        if (
            not isinstance(evidence_ids, list)
            or any(not isinstance(segment_id, str) or segment_id not in segment_ids for segment_id in evidence_ids)
        ):
            raise ValueError(
                f"{condition} feedback evidence must reference only segments in its reflection."
            )
        if EXPOSED_CONDITION_PATTERN.search(entry["text"]):
            raise ValueError(f"{condition} feedback text appears to disclose its model or condition.")
    return feedback


def _reference_scoring(rubric: dict[str, Any]) -> dict[str, Any]:
    """Evaluator-facing block for independent SW/UA/HA reference scoring (step 0)."""
    protocol = resolve_prompt(REFERENCE_SCORE_PROTOCOL_ID, REFERENCE_SCORE_PROTOCOL_VERSION)
    return {
        "protocol_id": protocol.artifact_id,
        "protocol_version": protocol.version,
        "human_protocol": protocol.text,
        "rubric_version": rubric["version"],
        "scale": {"min": 0, "max": 3, "step": 0.5},
        "dimensions": [
            {
                "dimension_id": dimension["dimension_id"],
                "name": dimension["name"],
                "description": dimension["description"],
                "bands": {band: dimension["bands"][band] for band in ("0", "1", "2", "3")},
            }
            for dimension in rubric["dimensions"]
        ],
    }


def _validate_reference_scoring(reference: dict[str, Any]) -> set[str]:
    if set(reference) != {"protocol_id", "protocol_version", "human_protocol", "rubric_version", "scale", "dimensions"}:
        raise ValueError("Reference scoring block contains unsupported fields.")
    if (
        reference["protocol_id"] != REFERENCE_SCORE_PROTOCOL_ID
        or reference["protocol_version"] != REFERENCE_SCORE_PROTOCOL_VERSION
        or not reference["human_protocol"].strip()
        or reference["scale"] != {"min": 0, "max": 3, "step": 0.5}
    ):
        raise ValueError("Reference scoring must use the current protocol and the 0-3 half-point scale.")
    dimension_ids = [dimension["dimension_id"] for dimension in reference["dimensions"]]
    if dimension_ids != ["SW", "UA", "HA"]:
        raise ValueError("Reference scoring must cover SW, UA and HA.")
    for dimension in reference["dimensions"]:
        if set(dimension) != {"dimension_id", "name", "description", "bands"} or set(dimension["bands"]) != {"0", "1", "2", "3"}:
            raise ValueError("Reference scoring dimensions need a name, description and anchors 0-3.")
    return set(dimension_ids)


def _validate_bundle(bundle: dict[str, Any]) -> None:
    if set(bundle) != {"schema_version", "protocol_id", "protocol_version", "reference_scoring", "cases"}:
        raise ValueError("Pairwise evaluator bundles may contain only the supported protocol fields.")
    if (
        bundle["schema_version"] != SCHEMA_VERSION
        or bundle["protocol_id"] != PROTOCOL_ID
        or bundle["protocol_version"] != SCHEMA_VERSION
        or not bundle["cases"]
    ):
        raise ValueError("Pairwise evaluator bundle has an unsupported or empty protocol.")
    reference_dimensions = _validate_reference_scoring(bundle["reference_scoring"])

    case_ids: set[str] = set()
    for case in bundle["cases"]:
        if set(case) != {
            "case_id", "reflection_id", "reflection", "candidates",
            "display_order", "human_annotations", "segment_summary",
        }:
            raise ValueError("Pairwise case contains fields outside the evaluator-facing schema.")
        if case["case_id"] in case_ids:
            raise ValueError("Pairwise evaluator bundle contains duplicate case IDs.")
        case_ids.add(case["case_id"])
        segment_ids = {
            segment["segment_id"] for segment in case["reflection"]["segments"]
        }
        candidates = case["candidates"]
        if len(candidates) != 3 or len({candidate["candidate_id"] for candidate in candidates}) != 3:
            raise ValueError("Every pairwise case must have exactly three distinct candidates.")
        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        if set(case["display_order"]) != {"A", "B", "C"} or set(case["display_order"].values()) != candidate_ids:
            raise ValueError("A/B/C display order must contain every candidate exactly once.")
        if {dimension["dimension_id"] for dimension in case["segment_summary"]["dimensions"]} != reference_dimensions:
            raise ValueError("Segment summary dimensions must match the reference scoring dimensions.")
        if "mean_positive_band" in json.dumps(case["segment_summary"]):
            raise ValueError("Descriptive segment information must not contain average scores.")
        for candidate in candidates:
            if not re.fullmatch(r"candidate-[0-9a-f]{16}", candidate["candidate_id"]):
                raise ValueError("Candidate IDs must be opaque, non-condition-revealing identifiers.")
            for component, entries in candidate["feedback"].items():
                if component not in {"strengths", "weaknesses", "suggestions"}:
                    raise ValueError("Feedback contains an unsupported component.")
                for entry in entries:
                    if set(entry) != {"text", "evidence_segment_ids"} or not entry["text"].strip():
                        raise ValueError("Feedback statements must have text and evidence IDs.")
                    if not set(entry["evidence_segment_ids"]) <= segment_ids:
                        raise ValueError("Feedback evidence refers to segments outside its reflection.")


def build_pairwise_evaluation_bundles(
    protected_root: str | Path,
    *,
    evaluator_ids: Sequence[str],
    cohort: Literal["playground", "remaining-test"] = "remaining-test",
    write: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build blinded v1.2 evaluator bundles (reference scoring + feedback preference) and a separate condition key."""
    if not evaluator_ids or any(
        not isinstance(value, str) or not EVALUATOR_ID_PATTERN.fullmatch(value)
        for value in evaluator_ids
    ):
        raise ValueError("Provide one or more pseudonymous evaluator IDs using letters, digits, dot, '_' or '-'.")
    if len(set(evaluator_ids)) != len(evaluator_ids):
        raise ValueError("Evaluator IDs must be unique.")

    config = load_experiment_config()
    root = Path(protected_root).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    if root == project_root or project_root in root.parents:
        raise ValueError("Protected pairwise packet storage must be outside the source tree.")

    generation = config["generation"]
    rubric_entry = resolve_rubric(generation["rubric_id"], generation["expected_rubric_version"])
    rubric = rubric_entry.rubric.model_dump(mode="json")
    split_path = (
        project_root
        / config["paths"]["provisional_split_directory"]
        / f"provisional_{config['dataset']['prediction_split']}_modeling_wide.csv"
    )
    rows = pd.read_csv(
        split_path,
        usecols=[
            "document_id", "segment_id", "candidate_id", "text", "segment_order",
            "scope_target", "scope_target_known",
            "target_situationserfassung", "known_situationserfassung",
            "target_analyse", "known_analyse",
            "target_konsequenzen", "known_konsequenzen", "requires_review",
        ],
        dtype={"document_id": str, "segment_id": str, "candidate_id": str},
    )
    cohort_id, selected_document_ids, _ = _select_packet_cohort(cohort, config=config)
    if not selected_document_ids or len(set(selected_document_ids)) != len(selected_document_ids):
        raise ValueError("Selected pairwise cohort is empty or contains duplicate document IDs.")
    if not set(selected_document_ids).issubset(set(rows["document_id"].dropna().astype(str))):
        raise ValueError("A selected pairwise document is missing from the configured test split.")

    g1_g3 = _load_latest_g1_g3_runs(
        root,
        document_ids=selected_document_ids,
        config=config,
        rubric_sha256=rubric_entry.sha256,
    )
    g2_runs = _load_author_disjoint_g2_runs(
        root,
        cohort=cohort,
        document_ids=selected_document_ids,
        config=config,
        project_root=project_root,
        rubric_sha256=rubric_entry.sha256,
    )
    reference_scoring = _reference_scoring(rubric)
    allocation_id = uuid4().hex
    permutation_order = list(itertools.permutations(CONDITIONS))
    cases: list[dict[str, Any]] = []
    key_records: list[dict[str, Any]] = []

    for case_index, document_id in enumerate(selected_document_ids):
        document_rows = rows.loc[rows["document_id"] == document_id].copy()
        segments = _reflection_segments(document_rows)
        reflection_id = make_reflection_id(segments)
        case_id = f"case-{hashlib.sha256(reflection_id.encode('utf-8')).hexdigest()[:16]}"
        segment_ids = {segment["segment_id"] for segment in segments}
        human_annotations = _human_annotations(document_rows, rubric)
        all_annotations = build_segment_modelling_annotations(document_rows)
        segment_summary = build_segment_annotation_summary(all_annotations)
        # The evaluator UI labels positive_band_count as a segment count.
        if any(
            dimension["positive_band_count"] != dimension["distinct_assigned_segment_count"]
            for dimension in segment_summary["dimensions"]
        ):
            raise ValueError("Each segment must have at most one assigned band per dimension for pairwise summaries.")
        comparison_path, comparison, pair = g1_g3[document_id]
        g2_path, g2_run = g2_runs[document_id]
        runs = {
            "G1": pair["G1"],
            "G2": g2_run,
            "G3": pair["G3"],
        }
        candidate_by_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            run = runs[condition]
            components = _validate_run_provenance(
                run,
                condition=condition,
                config=config,
                rubric_sha256=rubric_entry.sha256,
            )
            if str(run.get("document_id", document_id)) != document_id:
                raise ValueError(f"{condition} run document provenance does not match the reflection.")
            candidate_id = "candidate-" + hashlib.sha256(
                f"{allocation_id}:{reflection_id}:{condition}".encode("utf-8")
            ).hexdigest()[:16]
            candidate_by_condition[condition] = {
                "candidate_id": candidate_id,
                "feedback": _candidate_feedback(
                    run,
                    segment_ids=segment_ids,
                    condition=condition,
                ),
            }
            source_path = g2_path if condition == "G2" else comparison_path
            key_records.append(
                {
                    "case_id": case_id,
                    "candidate_id": candidate_id,
                    "condition": condition,
                    "document_id": document_id,
                    "run_id": run["run_id"],
                    "repetition": run["repetition"],
                    "source_artifact": source_path.name,
                    "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "prompt_components": components,
                }
            )

        cases.append(
            {
                "case_id": case_id,
                "reflection_id": reflection_id,
                "reflection": {"segments": segments},
                "candidates": [candidate_by_condition[condition] for condition in CONDITIONS],
                "display_order": {},
                "human_annotations": human_annotations,
                "segment_summary": segment_summary,
            }
        )

    evaluator_bundles: dict[str, dict[str, Any]] = {}
    ordered_evaluator_ids = sorted(evaluator_ids)
    for evaluator_index, evaluator_id in enumerate(ordered_evaluator_ids):
        evaluator_cases: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            order = permutation_order[
                (case_index * len(ordered_evaluator_ids) + evaluator_index) % len(permutation_order)
            ]
            candidate_ids = {
                record["condition"]: record["candidate_id"]
                for record in key_records
                if record["case_id"] == case["case_id"]
            }
            evaluator_cases.append(
                {
                    **case,
                    "display_order": {
                        label: candidate_ids[condition]
                        for label, condition in zip(("A", "B", "C"), order, strict=True)
                    },
                }
            )
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "protocol_version": SCHEMA_VERSION,
            "reference_scoring": reference_scoring,
            "cases": evaluator_cases,
        }
        _validate_bundle(bundle)
        evaluator_bundles[evaluator_id] = bundle

    condition_key = {
        "schema_version": SCHEMA_VERSION,
        "allocation_id": allocation_id,
        "cohort": cohort_id,
        "prompt_versions": {
            "generation": generation["expected_prompt_version"],
            "g2_supplemental": generation["g2_supplemental_prompt_version"],
            "g3_supplemental": generation["g3_supplemental_prompt_version"],
        },
        "rubric_version": generation["expected_rubric_version"],
        "reference_score_protocol": {
            "protocol_id": reference_scoring["protocol_id"],
            "protocol_version": reference_scoring["protocol_version"],
        },
        "experiment_config_sha256": experiment_config_sha256(),
        "records": key_records,
    }
    if write:
        output_root = root / "pairwise-evaluations" / "development"
        output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(output_root, 0o700)
        target_dir = output_root / cohort_id
        if target_dir.exists():
            raise FileExistsError("Refusing to overwrite existing pairwise evaluator bundles.")
        with tempfile.TemporaryDirectory(prefix=".pairwise-evaluations-", dir=output_root) as temporary:
            staging_dir = Path(temporary)
            os.chmod(staging_dir, 0o700)
            bundles_dir = staging_dir / "bundles"
            bundles_dir.mkdir(mode=0o700)
            for evaluator_id, bundle in evaluator_bundles.items():
                bundle_path = bundles_dir / f"{evaluator_id}.json"
                bundle_path.write_text(
                    json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.chmod(bundle_path, 0o600)
            key_path = staging_dir / "condition-key.json"
            key_path.write_text(
                json.dumps(condition_key, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.chmod(key_path, 0o600)
            os.rename(staging_dir, target_dir)
    return evaluator_bundles, condition_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build blinded pairwise feedback evaluation bundles in protected storage."
    )
    parser.add_argument("--protected-root", type=Path, default=os.environ.get("PREBI_DATA_ROOT"))
    parser.add_argument("--cohort", choices=("playground", "remaining-test"), default="remaining-test")
    parser.add_argument(
        "--evaluator-id",
        action="append",
        required=True,
        help="Pseudonymous evaluator ID; repeat once per expert.",
    )
    parser.add_argument("--write", action="store_true", help="Write bundles and the separate condition key.")
    arguments = parser.parse_args()
    if arguments.protected_root is None:
        parser.error("Set PREBI_DATA_ROOT or pass --protected-root.")
    try:
        bundles, _key = build_pairwise_evaluation_bundles(
            arguments.protected_root,
            evaluator_ids=arguments.evaluator_id,
            cohort=arguments.cohort,
            write=arguments.write,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    case_count = len(next(iter(bundles.values()))["cases"])
    print(
        f"Validated {case_count} cases for {len(bundles)} evaluators; "
        f"written={arguments.write}."
    )


if __name__ == "__main__":
    main()
