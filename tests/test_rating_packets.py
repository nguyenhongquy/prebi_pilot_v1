from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd

import reflection_assessment_feedback.rating_packets as rating_packets_module
from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.rating_packets import (
    build_rating_packet,
    build_reference_rating_packet,
    build_segment_annotation_summary,
    build_segment_modelling_annotations,
    make_reflection_id,
)
from reflection_assessment_feedback.prompt_registry import resolve_prompt, render_prompt
from reflection_assessment_feedback.rubric_registry import resolve_rubric

RUBRIC_ARTIFACT = resolve_rubric("unterrichtsreflexion-expert", "1.0.0")


def _segments() -> list[dict[str, object]]:
    return [{"segment_id": "segment-a", "order": 0, "text": "Synthetic reflection passage."}]


def _run() -> dict[str, object]:
    return {
        "run_id": "opaque-run-id",
        "condition": "G2",
        "repetition": 1,
        "model_name": "private-model-name",
        "output": {
            "assessment": {
                "component_assessments": [
                    {
                        "dimension_id": "SW",
                        "band": 2.1,
                        "justification": "Synthetic rationale.",
                        "evidence_segment_ids": ["segment-a"],
                    }
                ]
            },
            "feedback": {
                "strengths": [{"message": "Synthetic strength.", "evidence_segment_ids": ["segment-a"]}],
                "weaknesses": [],
                "suggestions": [],
            },
        },
    }


def test_assessment_and_feedback_packets_are_blinded_and_task_specific() -> None:
    config = load_experiment_config()["prompts"]
    rubric = RUBRIC_ARTIFACT.rubric.model_dump(mode="json")
    assessment_packet = build_rating_packet(
        task="assessment_quality",
        run=_run(),
        reflection_segments=_segments(),
        rubric=rubric,
    )
    feedback_packet = build_rating_packet(
        task="feedback_quality",
        run=_run(),
        reflection_segments=_segments(),
        rubric=rubric,
        human_feedback="Synthetic historical lecturer feedback.",
    )

    for packet in (assessment_packet, feedback_packet):
        assert packet["condition_blinded"] is True
        assert "condition" not in packet
        assert "model_name" not in packet
        assert "document_id" not in packet
        assert packet["human_protocol"]
        assert packet["displayed_reflection_sha256"]
        assert packet["display_output_sha256"]

    assert assessment_packet["task"] == "assessment_quality"
    assert assessment_packet["protocol_id"] == config["assessment_quality_human_id"]
    assert "dimensions" in assessment_packet["output"]
    assert assessment_packet["displayed_context"] == [
        "segmented_reflection", "rubric", "assessment_output"
    ]
    assert feedback_packet["task"] == "feedback_quality"
    assert feedback_packet["protocol_id"] == config["feedback_quality_human_id"]
    assert feedback_packet["human_feedback"] == "Synthetic historical lecturer feedback."
    assert feedback_packet["output"]["strengths"][0]["text"] == "Synthetic strength."
    for packet in (assessment_packet, feedback_packet):
        assert packet["scale"]["min"] == 1
        assert packet["scale"]["max"] == 3
        for criterion in packet["criteria"]:
            assert set(criterion["anchors"]) == {"1", "2", "3"}
        assert any(character in packet["human_protocol"] for character in "äöüÄÖÜ")
    assert [criterion["label"] for criterion in feedback_packet["criteria"]] == [
        "Korrektheit", "Entwicklungsnutzen"
    ]
    assert [criterion["criterion_id"] for criterion in feedback_packet["criteria"]] == [
        "correctness", "developmental_usefulness"
    ]
    assert feedback_packet["protocol_version"] == "0.3.0"
    assert assessment_packet["protocol_version"] == "0.4.0"
    assert [criterion["label"] for criterion in assessment_packet["criteria"]] == [
        "Passung von Punktwert und Rubrik", "Belegstützung", "Qualität der Begründung"
    ]
    assert [criterion["criterion_id"] for criterion in assessment_packet["criteria"]] == [
        "score_rubric_fit", "evidence_support", "justification_quality"
    ]
    with pytest.raises(ValueError, match="not permitted on Task B"):
        build_rating_packet(
            task="assessment_quality",
            run=_run(),
            reflection_segments=_segments(),
            rubric=rubric,
            human_feedback="Must stay out of Task B.",
        )


@pytest.mark.parametrize("task", ["assessment_quality", "feedback_quality"])
def test_packet_rejects_evidence_from_another_reflection(task: str) -> None:
    with pytest.raises(ValueError, match="evidence.*displayed reflection"):
        build_rating_packet(
            task=task,
            run=_run(),
            reflection_segments=[{"segment_id": "other-segment", "order": 0, "text": "Other reflection."}],
            rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
        )


def test_bundle_uses_each_documents_g2_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config()
    config["generation"]["expected_rubric_version"] = "1.0.0"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    config["paths"]["provisional_split_directory"] = str(split_dir)
    config["dataset"]["prediction_split"] = "test"
    monkeypatch.setattr(rating_packets_module, "load_experiment_config", lambda: config)
    monkeypatch.setattr(
        rating_packets_module, "_select_packet_cohort",
        lambda cohort, **kwargs: ("synthetic-cohort", ["a", "b"], {"a": "Reference A", "b": "Reference B"}),
    )
    comparison_dir = tmp_path / "g1-g3" / "demo-runs"
    comparison_dir.mkdir(parents=True)
    rows = []
    g2_runs = {}
    for document_id in ("a", "b"):
        rows.append({
            "document_id": document_id, "segment_id": f"segment-{document_id}",
            "candidate_id": f"candidate-{document_id}", "text": f"Reflection {document_id}",
            "segment_order": 0, "scope_target": 1, "scope_target_known": True,
            "target_situationserfassung": 2, "known_situationserfassung": True,
            "target_analyse": 2, "known_analyse": True,
            "target_konsequenzen": 2, "known_konsequenzen": True, "requires_review": False,
        })
        runs = {}
        for condition in ("G1", "G2", "G3"):
            run = _run()
            run.update({
                "run_id": f"{condition}-{document_id}",
                "condition": condition,
                "model_name": config["generation"]["model_name"],
                "prompt_version": config["generation"]["expected_prompt_version"],
                "repetition": config["generation"]["repetitions"],
                "generation_parameters": {},
            })
            run["output"]["assessment"]["component_assessments"][0]["evidence_segment_ids"] = [f"segment-{document_id}"]
            run["output"]["feedback"]["strengths"][0].update({
                "message": f"Strength {document_id}",
                "evidence_segment_ids": [f"segment-{document_id}"],
            })
            runs[condition] = run
        comparison = {
            "document_id": document_id,
            "rubric_id": config["generation"]["rubric_id"],
            "rubric_version": config["generation"]["expected_rubric_version"],
            "pairs": [{"G1": runs["G1"], "G3": runs["G3"]}],
        }
        (comparison_dir / f"comparison-{document_id}.json").write_text(json.dumps(comparison))
        g2_path = tmp_path / f"g2-{document_id}.json"
        g2_path.write_text(json.dumps(runs["G2"]))
        g2_runs[document_id] = (g2_path, runs["G2"])
    pd.DataFrame(rows).to_csv(split_dir / "provisional_test_modeling_wide.csv", index=False)
    monkeypatch.setattr(rating_packets_module, "_load_author_disjoint_g2_runs", lambda *args, **kwargs: g2_runs)

    bundle, key = rating_packets_module.build_development_rating_packet_bundle(tmp_path, write=True)
    packets = {packet["packet_id"]: packet for packet in bundle["packets"]}
    assert len(packets) == 14
    for record in key["records"]:
        if record["condition"] != "G2":
            continue
        document_id = record["document_id"]
        assert record["run_id"] == f"G2-{document_id}"
        assert record["source_artifact_sha256"] == hashlib.sha256(g2_runs[document_id][0].read_bytes()).hexdigest()
        packet = packets[record["packet_id"]]
        if record["task"] == "feedback_quality":
            assert packet["output"]["strengths"][0]["text"] == f"Strength {document_id}"
        else:
            assert packet["output"]["dimensions"][0]["evidence_segment_ids"] == [f"segment-{document_id}"]
    output_dir = tmp_path / "rating-packets" / "development"
    assert json.loads((output_dir / "synthetic-cohort-rating-packets.json").read_text()) == bundle
    assert json.loads((output_dir / "synthetic-cohort-condition-key.json").read_text()) == key
    with pytest.raises(FileExistsError, match="overwrite"):
        rating_packets_module.build_development_rating_packet_bundle(tmp_path, write=True)


def test_reference_packet_is_independent_and_shares_stable_reflection_identity() -> None:
    rubric = RUBRIC_ARTIFACT.rubric.model_dump(mode="json")
    segments = _segments()
    reflection_id = make_reflection_id(segments)
    reference_packet = build_reference_rating_packet(
        reflection_segments=segments,
        rubric=rubric,
        reflection_id=reflection_id,
    )
    assessment_packet = build_rating_packet(
        task="assessment_quality",
        run=_run(),
        reflection_segments=segments,
        rubric=rubric,
        reflection_id=reflection_id,
    )

    assert make_reflection_id(segments) == reflection_id
    assert reference_packet["reflection_id"] == assessment_packet["reflection_id"]
    assert reference_packet["task"] == "reflection_reference_score"
    assert reference_packet["protocol_id"] == "reflection_reference_score_human"
    assert reference_packet["protocol_version"] == "0.2.0"
    protocol_text = reference_packet["human_protocol"]
    assert "Kommentar (optional)" in protocol_text
    assert "not averaged" in protocol_text
    assert "evidence links" not in protocol_text and "consensus" not in protocol_text
    assert reference_packet["scale"] == {
        "min": 0,
        "max": 3,
        "step": 0.5,
        "id": "reflection-reference-0-3-step-0.5-v1",
        "anchors": {"0": "Rubrikanker 0", "1": "Rubrikanker 1", "2": "Rubrikanker 2", "3": "Rubrikanker 3"},
    }
    assert reference_packet["score_dimensions"] == [
        {"dimension_id": dimension["dimension_id"], "label": dimension["name"]}
        for dimension in rubric["dimensions"]
    ]
    assert reference_packet["reflection"] == {"segments": segments}
    assert set(reference_packet["rubric"]) == {"version", "dimensions"}
    assert all(set(dimension) == {"dimension_id", "name", "description", "bands"} for dimension in reference_packet["rubric"]["dimensions"])
    assert reference_packet["condition_blinded"] is True
    assert "output" not in reference_packet
    assert "human_feedback" not in reference_packet
    assert "document_id" not in reference_packet


def test_reference_packet_preserves_modelling_candidates_and_unknown_dimension_values() -> None:
    rows = pd.DataFrame([
        {
            "segment_id": "segment-a", "candidate_id": "candidate-a", "scope_target": 1,
            "scope_target_known": True, "target_situationserfassung": 2,
            "known_situationserfassung": True, "target_analyse": float("nan"),
            "known_analyse": False, "target_konsequenzen": 1,
            "known_konsequenzen": True, "requires_review": False,
        },
        {
            "segment_id": "segment-a", "candidate_id": "candidate-b", "scope_target": 0,
            "scope_target_known": True, "target_situationserfassung": "N",
            "known_situationserfassung": False, "target_analyse": "N",
            "known_analyse": False, "target_konsequenzen": "N",
            "known_konsequenzen": False, "requires_review": True,
        },
    ])

    annotations = build_segment_modelling_annotations(rows)
    packet = build_reference_rating_packet(
        reflection_segments=_segments(),
        rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
        reflection_id=make_reflection_id(_segments()),
        segment_annotations=annotations,
    )

    assert len(packet["segment_annotations"]) == 6
    first_candidate = [item for item in packet["segment_annotations"] if item["candidate_id"] == "candidate-a"]
    second_candidate = [item for item in packet["segment_annotations"] if item["candidate_id"] == "candidate-b"]
    assert len(first_candidate) == len(second_candidate) == 3
    assert {item["scope"] for item in first_candidate} == {"in_scope"}
    assert {item["scope"] for item in second_candidate} == {"out_of_scope"}
    assert next(item for item in first_candidate if item["dimension_id"] == "UA")["known"] is False
    assert next(item for item in first_candidate if item["dimension_id"] == "UA")["level"] == ""
    assert all(item["requires_review"] for item in second_candidate)


def test_modelling_band_summary_counts_candidates_and_distinct_segments_without_means() -> None:
    annotations = [
        {"dimension_id": "SW", "level": "0", "known": True},
        {"dimension_id": "SW", "level": "1", "known": True},
        {"dimension_id": "SW", "level": "3", "known": True},
        {"dimension_id": "SW", "level": "2", "known": False},
        {"dimension_id": "UA", "level": "0", "known": True},
        {"dimension_id": "UA", "level": "2", "known": True},
        {"dimension_id": "HA", "level": "N", "known": False},
    ]
    for index, annotation in enumerate(annotations):
        annotation["segment_id"] = f"segment-{index}"

    summary = build_segment_annotation_summary(annotations)
    by_dimension = {item["dimension_id"]: item for item in summary["dimensions"]}

    assert summary["aggregation_unit"] == "candidate_annotation"
    assert by_dimension["SW"]["known_band_count"] == 3
    assert by_dimension["SW"]["band_counts"] == {"0": 1, "1": 1, "2": 0, "3": 1}
    assert by_dimension["SW"]["band_percentages"] == {"1": 50.0, "2": 0.0, "3": 50.0}
    assert by_dimension["SW"]["positive_band_count"] == 2
    assert by_dimension["SW"]["distinct_assigned_segment_count"] == 2
    assert by_dimension["UA"]["distinct_assigned_segment_count"] == 1
    assert by_dimension["HA"]["known_band_count"] == 0
    assert all("mean_positive_band" not in item for item in summary["dimensions"])


def test_playground_rating_packets_exclude_document_126(monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = pd.DataFrame([
        {"document_id": "3", "feedback_text": "feedback 3"},
        {"document_id": "64", "feedback_text": "feedback 64"},
        {"document_id": "126", "feedback_text": "feedback 126"},
        {"document_id": "188", "feedback_text": "feedback 188"},
    ])
    monkeypatch.setattr(rating_packets_module, "load_playground_feedback_cohort", lambda: cohort)

    cohort_id, document_ids, feedback = rating_packets_module._select_packet_cohort(
        "playground", config={}
    )

    assert cohort_id == "playground-author-188-rubric-1-2-0-pipeline-2"
    assert document_ids == ["3", "64", "188"]
    assert feedback == {"3": "feedback 3", "64": "feedback 64", "188": "feedback 188"}


def test_task_c_packet_builder_carries_historical_feedback_only_for_feedback_quality() -> None:
    rubric = RUBRIC_ARTIFACT.rubric.model_dump(mode="json")
    feedback_packet = build_rating_packet(
        task="feedback_quality", run=_run(), reflection_segments=_segments(),
        rubric=rubric, reflection_id="reflection-test", human_feedback="Historical lecturer feedback.",
    )
    assert feedback_packet["human_feedback"] == "Historical lecturer feedback."
    assert feedback_packet["reflection_id"] == "reflection-test"
    with pytest.raises(ValueError, match="not permitted on Task B"):
        build_rating_packet(
            task="assessment_quality", run=_run(), reflection_segments=_segments(),
            rubric=rubric, human_feedback="Historical lecturer feedback.",
        )


def test_legacy_quality_protocol_keeps_original_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reflection_assessment_feedback.rating_packets.QUALITY_HUMAN_PROTOCOL_VERSIONS",
        {"assessment_quality": "0.4.0", "feedback_quality": "0.1.0"},
    )
    packet = build_rating_packet(
        task="feedback_quality", run=_run(), reflection_segments=_segments(),
        rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
    )
    assert packet["protocol_version"] == "0.1.0"
    assert packet["scale"]["max"] == 5
    assert len(packet["criteria"]) == 5


@pytest.mark.parametrize("task", ["assessment_quality", "feedback_quality"])
def test_revised_quality_llm_prompts_are_registered_and_renderable(task: str) -> None:
    config = load_experiment_config()["prompts"]
    artifact = resolve_prompt(config[f"{task}_llm_id"], config[f"{task}_llm_version"])
    rendered, digest = render_prompt(artifact, packet_json="{}")
    assert artifact.version == ("0.5.0" if task == "assessment_quality" else "0.3.0")
    assert "Referenzfeedback" in artifact.text
    assert "Beurteile" in artifact.text
    if task == "assessment_quality":
        assert "Gold-Segmentanalyse" in artifact.text
    assert "$packet_json" not in rendered
    assert len(digest) == 64


def test_assessment_v02_preserves_four_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reflection_assessment_feedback.rating_packets.QUALITY_HUMAN_PROTOCOL_VERSIONS",
        {"assessment_quality": "0.2.0", "feedback_quality": "0.3.0"},
    )
    packet = build_rating_packet(
        task="assessment_quality", run=_run(), reflection_segments=_segments(),
        rubric=RUBRIC_ARTIFACT.rubric.model_dump(mode="json"),
    )
    assert packet["protocol_version"] == "0.2.0"
    assert [criterion["criterion_id"] for criterion in packet["criteria"]] == [
        "rubric_alignment", "score_defensibility", "evidence_support", "justification_quality"
    ]
    assert packet["scale"]["max"] == 3
