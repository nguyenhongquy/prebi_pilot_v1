from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import reflection_assessment_feedback.pairwise_packets as pairwise_packets
from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.prompt_registry import resolve_prompt
from reflection_assessment_feedback.rubric_registry import resolve_rubric


def _run(condition: str, config: dict[str, Any], rubric_sha256: str) -> dict[str, Any]:
    generation = config["generation"]
    base_prompt = resolve_prompt(generation["prompt_id"], generation["expected_prompt_version"])
    components = [{
        "artifact_id": base_prompt.artifact_id,
        "version": base_prompt.version,
        "sha256": base_prompt.sha256,
    }]
    if condition == "G2":
        supplemental = resolve_prompt(
            generation["g2_supplemental_prompt_id"],
            generation["g2_supplemental_prompt_version"],
        )
        components.append({
            "artifact_id": supplemental.artifact_id,
            "version": supplemental.version,
            "sha256": supplemental.sha256,
        })
    elif condition == "G3":
        supplemental = resolve_prompt(
            generation["g3_supplemental_prompt_id"],
            generation["g3_supplemental_prompt_version"],
        )
        components.append({
            "artifact_id": supplemental.artifact_id,
            "version": supplemental.version,
            "sha256": supplemental.sha256,
        })
    return {
        "run_id": f"opaque-{condition.lower()}-run",
        "condition": condition,
        "document_id": "doc-x",
        "repetition": generation["repetitions"],
        "model_name": generation["model_name"],
        "generation_parameters": {},
        "prompt_version": base_prompt.version,
        "prompt_artifact_sha256": base_prompt.sha256,
        "prompt_components": components,
        "pipeline_version": generation["pipeline_version"],
        "rubric_version": generation["expected_rubric_version"],
        "rubric_artifact_sha256": rubric_sha256,
        "output": {
            "feedback": {
                "strengths": [{
                    "message": "A specific evidence-linked feedback statement.",
                    "evidence_segment_ids": ["segment-x"],
                }],
                "weaknesses": [],
                "suggestions": [],
            }
        },
    }


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    config = load_experiment_config()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    config["paths"]["provisional_split_directory"] = str(split_dir)
    config["dataset"]["prediction_split"] = "test"
    monkeypatch.setattr(pairwise_packets, "load_experiment_config", lambda: config)
    monkeypatch.setattr(
        pairwise_packets,
        "_select_packet_cohort",
        lambda cohort, **kwargs: ("synthetic-cohort", ["doc-x"], {}),
    )
    rows = pd.DataFrame([{
        "document_id": "doc-x",
        "segment_id": "segment-x",
        "candidate_id": "human-candidate-x",
        "text": "A synthetic reflection.",
        "segment_order": 0,
        "scope_target": 1,
        "scope_target_known": True,
        "target_situationserfassung": 2,
        "known_situationserfassung": True,
        "target_analyse": 1,
        "known_analyse": True,
        "target_konsequenzen": 0,
        "known_konsequenzen": True,
        "requires_review": False,
    }])
    rows.to_csv(split_dir / "provisional_test_modeling_wide.csv", index=False)

    rubric = resolve_rubric(
        config["generation"]["rubric_id"],
        config["generation"]["expected_rubric_version"],
    )
    runs = {
        condition: _run(condition, config, rubric.sha256)
        for condition in ("G1", "G2", "G3")
    }
    comparison_dir = tmp_path / "g1-g3" / "demo-runs"
    comparison_dir.mkdir(parents=True)
    comparison_path = comparison_dir / "comparison-20261014T112715Z-synthetic.json"
    comparison_path.write_text(
        json.dumps({
            "document_id": "doc-x",
            "rubric_id": config["generation"]["rubric_id"],
            "rubric_version": config["generation"]["expected_rubric_version"],
            "pairs": [{
                "repetition": config["generation"]["repetitions"],
                "G1": runs["G1"],
                "G3": runs["G3"],
            }],
        }),
        encoding="utf-8",
    )
    g2_path = tmp_path / "g2-run.json"
    g2_path.write_text(json.dumps({"run": runs["G2"]}), encoding="utf-8")
    monkeypatch.setattr(
        pairwise_packets,
        "_load_author_disjoint_g2_runs",
        lambda *args, **kwargs: {"doc-x": (g2_path, runs["G2"])},
    )
    return config, tmp_path, runs


def test_pairwise_bundles_are_blinded_counterbalanced_and_versioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protected_root, _ = _setup(tmp_path, monkeypatch)
    evaluator_ids = [f"expert-{index}" for index in range(6)]

    bundles, key = pairwise_packets.build_pairwise_evaluation_bundles(
        protected_root,
        evaluator_ids=evaluator_ids,
        write=True,
    )

    assert set(bundles) == set(evaluator_ids)
    assert all(bundle["schema_version"] == "1.2.0" for bundle in bundles.values())
    reference = next(iter(bundles.values()))["reference_scoring"]
    assert reference["protocol_id"] == "reflection_reference_score_human"
    assert reference["protocol_version"] == "0.2.0"
    assert "Kommentar (optional)" in reference["human_protocol"]
    assert reference["scale"] == {"min": 0, "max": 3, "step": 0.5}
    assert [dimension["dimension_id"] for dimension in reference["dimensions"]] == ["SW", "UA", "HA"]
    assert all(set(dimension["bands"]) == {"0", "1", "2", "3"} for dimension in reference["dimensions"])
    assert key["reference_score_protocol"] == {
        "protocol_id": "reflection_reference_score_human", "protocol_version": "0.2.0",
    }
    assert all(len(bundle["cases"]) == 1 for bundle in bundles.values())
    first_case = next(iter(bundles.values()))["cases"][0]
    assert len(first_case["candidates"]) == 3
    assert [item["skill_label"] for item in first_case["human_annotations"]] == [
        "Handlungsalternativen", "Situationswahrnehmung", "Ursachenanalyse",
    ]
    assert {item["level"] for item in first_case["human_annotations"]} == {"Band 0", "Band 1", "Band 2"}
    assert all("mean_positive_band" not in dimension for dimension in first_case["segment_summary"]["dimensions"])

    all_orders = {
        tuple(
            next(candidate for candidate in bundle["cases"][0]["candidates"]
                 if candidate["candidate_id"] == bundle["cases"][0]["display_order"][label])["candidate_id"]
            for label in ("A", "B", "C")
        )
        for bundle in bundles.values()
    }
    assert len(all_orders) == 6
    assert all(
        not any(condition in candidate["candidate_id"] for condition in ("G1", "G2", "G3"))
        for bundle in bundles.values()
        for candidate in bundle["cases"][0]["candidates"]
    )
    assert {record["condition"] for record in key["records"]} == {"G1", "G2", "G3"}
    assert all(
        record["prompt_components"] == pairwise_packets._expected_prompt_components(
            load_experiment_config(), record["condition"]
        )
        for record in key["records"]
    )

    output_dir = protected_root / "pairwise-evaluations" / "development" / "synthetic-cohort"
    assert json.loads((output_dir / "bundles" / "expert-0.json").read_text()) == bundles["expert-0"]
    saved_key = json.loads((output_dir / "condition-key.json").read_text())
    assert saved_key == key
    assert stat.S_IMODE((output_dir / "bundles" / "expert-0.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output_dir / "condition-key.json").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError, match="overwrite"):
        pairwise_packets.build_pairwise_evaluation_bundles(
            protected_root,
            evaluator_ids=evaluator_ids,
            write=True,
        )


def test_pairwise_source_with_noncurrent_prompt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, protected_root, runs = _setup(tmp_path, monkeypatch)
    comparison_path = protected_root / "g1-g3" / "demo-runs" / "comparison-20261014T112715Z-synthetic.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["pairs"][0]["G3"]["prompt_version"] = "1.4.0"
    comparison["pairs"][0]["G3"]["prompt_components"][0]["version"] = "1.4.0"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="latest active prompt"):
        pairwise_packets.build_pairwise_evaluation_bundles(
            protected_root,
            evaluator_ids=["expert-1"],
        )


def test_human_annotations_keep_review_status_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, protected_root, _ = _setup(tmp_path, monkeypatch)
    rows = pd.read_csv(
        protected_root / "splits" / "provisional_test_modeling_wide.csv",
        dtype={"document_id": str, "segment_id": str, "candidate_id": str},
    )
    rows["requires_review"] = True
    rubric = resolve_rubric(
        config["generation"]["rubric_id"],
        config["generation"]["expected_rubric_version"],
    ).rubric.model_dump(mode="json")

    annotations = pairwise_packets._human_annotations(rows, rubric)

    assert annotations
    assert all(annotation["level"].endswith("(prüfbedürftig)") for annotation in annotations)


def test_statement_without_cited_segment_is_kept_but_unknown_segments_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protected_root, runs = _setup(tmp_path, monkeypatch)
    absent_content = {"message": "The reflection does not discuss alternatives.", "evidence_segment_ids": []}
    runs["G2"]["output"]["feedback"]["weaknesses"] = [absent_content]

    bundles, key = pairwise_packets.build_pairwise_evaluation_bundles(protected_root, evaluator_ids=["expert-0"])

    g2_candidate = next(record["candidate_id"] for record in key["records"] if record["condition"] == "G2")
    candidate = next(
        item for item in bundles["expert-0"]["cases"][0]["candidates"] if item["candidate_id"] == g2_candidate
    )
    assert candidate["feedback"]["weaknesses"] == [
        {"text": absent_content["message"], "evidence_segment_ids": []}
    ]

    absent_content["evidence_segment_ids"] = ["segment-from-another-reflection"]
    with pytest.raises(ValueError, match="only segments in its reflection"):
        pairwise_packets.build_pairwise_evaluation_bundles(protected_root, evaluator_ids=["expert-0"])
